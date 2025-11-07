import numpy as np
import torch


class PushingCostFunctions():
    def __init__(self, device='cpu', weights=None):
        self.trajectory = None
        self.target_waypoint = None
        self.waypoint_idx = None
        self.goal = None
        self.device = device
        self.set_weights(weights)
        self.original_trajectory = None
        self.waypoint_lookahead = 0.16
        self.threshold = 0.08
        self.index = 0
        self.sample_null_ = False
        self.replan = False
        self.terminal_scale = 100.0

    def set_weights(self, weights):
        if weights is not None:
            self.W1 = weights[0]
        else:
            self.W1 = 1

    def set_trajectory(self, trajectory):
        self.goal = trajectory[-1]
        self.N = trajectory.shape[0]
        self.trajectory = trajectory

    def get_reference_index(self, obs):
        block_pose = obs[3:5]  # block x, y
        block_pose = np.array(block_pose)

        diff = self.trajectory[:, :2] - block_pose[:2]
        dist = np.linalg.norm(diff[:, :2], axis=1)
        index = dist.argmin()

        while (dist[index] < self.waypoint_lookahead and index <= len(self.trajectory) - 2):
            index += 1
            index = min(index, len(self.trajectory) - 1)
        if (len(self.trajectory) == 1):
            self.index = 0
            return 0
        if dist[self.N - 1] <= self.threshold and index >= self.N - 1:
            self.N = self.trajectory.shape[0]
            index = 0
        if np.linalg.norm(self.goal[:2] - block_pose[:2]) < 0.15:
            print("very close to goal, sampling null poses now")
            self.sample_null_ = True
        self.index = index
        return index

    def sample_null(self):
        return self.sample_null_

    def tan_dist(self, poses, trajectory):
        """
        Compute minimum distance from poses to trajectory segments.
        Works with both numpy arrays and torch tensors.
        """
        # Convert to torch if needed
        is_torch = isinstance(poses, torch.Tensor)
        if not is_torch:
            poses = torch.tensor(poses, dtype=torch.float32)

        # Convert trajectory to torch
        if isinstance(trajectory, np.ndarray):
            trajectory = torch.tensor(trajectory, dtype=torch.float32, device=poses.device)

        N = poses.shape[0]
        M = trajectory.shape[0]

        A = trajectory[:-1, :2]  # (M-1) x 2
        B = trajectory[1:, :2]  # (M-1) x 2
        AB = B - A  # (M-1) x 2
        AB_norm_sq = torch.sum(AB ** 2, dim=1)  # (M-1)

        # Handle zero-length segments
        zero_length = AB_norm_sq == 0
        distances = torch.full((N,), float('inf'), device=poses.device)

        if torch.any(zero_length):
            A_zero = A[zero_length]  # K x 2
            diff = poses[:, None, :2] - A_zero[None, :, :]  # N x K x 2
            dist_sq = torch.sum(diff ** 2, dim=2)  # N x K
            dist_zero = torch.sqrt(dist_sq)  # N x K
            distances = torch.minimum(distances, torch.min(dist_zero, dim=1)[0])

        # Handle non-zero-length segments
        if torch.any(~zero_length):
            A_nonzero = A[~zero_length]  # S x 2
            AB_nonzero = AB[~zero_length]  # S x 2
            AB_norm_sq_nonzero = AB_norm_sq[~zero_length]  # S

            AP = poses[:, None, :2] - A_nonzero[None, :, :]  # N x S x 2
            numerator = torch.einsum('nsi,si->ns', AP, AB_nonzero)  # N x S
            denominator = AB_norm_sq_nonzero  # S
            t = numerator / denominator[None, :]  # N x S
            t = torch.clamp(t, 0, 1)  # Clamp t to [0, 1]

            C = A_nonzero[None, :, :] + t[:, :, None] * AB_nonzero[None, :, :]  # N x S x 2
            diff = poses[:, None, :2] - C  # N x S x 2
            dist_sq = torch.sum(diff ** 2, dim=2)  # N x S
            dist = torch.sqrt(dist_sq)  # N x S
            distances = torch.minimum(distances, torch.min(dist, dim=1)[0])

        # Convert back to numpy if input was numpy
        if not is_torch:
            distances = distances.cpu().numpy()

        return distances

    def running_cost(self, states, actions):
        """
        Cost function based on BLOCK trajectory tracking.
        states: [car_x, car_y, car_theta, block_x, block_y, block_theta]
        """
        # Ensure states and actions are torch tensors
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.float32)

        # Extract BLOCK states (indices 3, 4, 5)
        block = states[:, 3:6]  # [block_x, block_y, block_theta]

        # Convert trajectory reference to tensor
        traj_ref = torch.tensor(self.trajectory[self.index], dtype=torch.float32, device=states.device)

        # Block tracking costs
        angle_diff = block[:, 2] - traj_ref[2]
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi

        position_cost_x = 1 * (block[:, 0] - traj_ref[0]) ** 2
        position_cost_y = 1 * (block[:, 1] - traj_ref[1]) ** 2
        heading_cost = 2 * (angle_diff) ** 2

        target_cost = position_cost_x + position_cost_y + heading_cost

        # Heading velocity cost (for smoother pushing)
        if self.index < len(self.trajectory) - 1:
            target_angular_vel = self.trajectory[self.index + 1, 2] - self.trajectory[self.index, 2]
            target_angular_vel = ((target_angular_vel + np.pi) % (2 * np.pi)) - np.pi
            heading_velocity_cost = 5 * abs(target_angular_vel) * (angle_diff) ** 2
            target_cost += heading_velocity_cost

        # Distance from block to trajectory path
        traj_distances = self.tan_dist(block[:, :2], self.trajectory[:, :2])
        traj_cost = 1.5 * traj_distances ** 2

        # Action costs (penalize large control inputs)
        action_cost_throttle = 0.001 * actions[:, 0] ** 2
        action_cost_steering = 0.01 * actions[:, 1] ** 2
        action_cost = action_cost_throttle + action_cost_steering

        # Optional: Add cost for car-block alignment to encourage better pushing posture
        car = states[:, :3]
        car_to_block_angle = torch.atan2(block[:, 1] - car[:, 1], block[:, 0] - car[:, 0])
        alignment_error = car[:, 2] - car_to_block_angle
        alignment_error = ((alignment_error + np.pi) % (2 * np.pi)) - np.pi
        alignment_cost = 0.5 * alignment_error ** 2  # Encourage car to face block

        cost = target_cost + traj_cost + action_cost + alignment_cost

        return cost

    def terminal_state_cost(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Terminal cost based on BLOCK final state.
        """
        final_states = s[0, :, -1, :]  # shape: NUM_SAMPLES x 6

        # Extract final block states
        final_block = final_states[:, 3:6]  # [block_x, block_y, block_theta]

        goal_position = torch.tensor(self.trajectory[-1, :2], dtype=torch.float32, device=s.device)
        goal_heading = self.trajectory[-1, 2]

        # Block position error at goal
        position_cost = torch.sum((final_block[:, :2] - goal_position) ** 2, dim=1)

        # Block heading error at goal
        angle_diff = final_block[:, 2] - goal_heading
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        heading_cost = 1.5 * (angle_diff) ** 2

        terminal_cost = position_cost + heading_cost
        return terminal_cost

    def push_dynamics(self, states, actions):
        dt = 0.01
        min_push_velocity = 0.2  # m/s - minimum velocity to overcome static friction

        # Extract current states
        x_now = states[:, 0]
        y_now = states[:, 1]
        Th_now = states[:, 2]
        block_x = states[:, 3]
        block_y = states[:, 4]
        block_heading = states[:, 5]

        # Extract actions
        steering_angle = actions[:, 0]
        speed = actions[:, 1]

        # Car dynamics
        x_dot = speed * torch.cos(Th_now) * dt
        y_dot = speed * torch.sin(Th_now) * dt
        theta_dot = ((speed * torch.tan(steering_angle)) / 0.295) * dt
        is_pushing = torch.abs(speed) >= min_push_velocity

        x_next = x_now + x_dot
        y_next = y_now + y_dot
        Th_next = Th_now + theta_dot

        # Compute current offset from car to block in GLOBAL frame
        dx_global = block_x - x_now
        dy_global = block_y - y_now

        # Transform offset to CAR's current frame
        # Rotation matrix from global to car frame
        cos_th = torch.cos(Th_now)
        sin_th = torch.sin(Th_now)

        # Offset in car's frame
        offset_x_car = cos_th * dx_global + sin_th * dy_global
        offset_y_car = -sin_th * dx_global + cos_th * dy_global

        # The offset in car's frame is maintained during pushing
        # Transform the maintained offset back to GLOBAL frame using car's NEXT heading
        cos_th_next = torch.cos(Th_next)
        sin_th_next = torch.sin(Th_next)

        # New block position = car's next position + offset in car's next frame transformed to global
        block_x_push = x_next + (cos_th_next * offset_x_car - sin_th_next * offset_y_car)
        block_y_push = y_next + (sin_th_next * offset_x_car + cos_th_next * offset_y_car)

        # Block heading rotates with the car (maintaining relative orientation)
        heading_offset = block_heading - Th_now  # relative heading in current frame
        block_heading_push = Th_next + heading_offset  # maintain relative heading

        # Apply dynamics based on whether we're pushing or not
        block_x_next = torch.where(
            is_pushing,
            block_x_push,
            block_x
        )

        block_y_next = torch.where(
            is_pushing,
            block_y_push,
            block_y
        )

        block_theta_next = torch.where(
            is_pushing,
            block_heading_push,
            block_heading
        )

        # Car also doesn't move if not pushing (stuck behind block)
        x_next = torch.where(is_pushing, x_next, x_now)
        y_next = torch.where(is_pushing, y_next, y_now)
        Th_next = torch.where(is_pushing, Th_next, Th_now)

        next_states = torch.stack(
            (x_next, y_next, Th_next, block_x_next, block_y_next, block_theta_next),
            dim=1
        )

        return next_states