import numpy as np
import torch

class CarCostFunctions():
    def __init__(self, device='cpu', weights=None):
        self.trajectory = None
        self.target_waypoint = None
        self.waypoint_idx = None
        self.goal = None
        self.device = device
        self.set_weights(weights)
        self.original_trajectory = None
        self.waypoint_lookahead = 0.08
        self.threshold = self.waypoint_lookahead - 0.05
        self.index = 0
        self.change_dir = False
        self.sample_null_ = False
        self.start_forward = True
        self.replan = False
        self.terminal_scale = 100.0

    def set_weights(self, weights):
        if weights is not None:
            self.W1 = weights[0]
        else:
            self.W1 = 1

    def check_forward(self, pt1, pt2, o):
        v = pt2 - pt1
        d = np.array([np.cos(o), np.sin(o)])
        dp = np.dot(v, d)
        if dp >= 0:
            forward = True
        else:
            forward = False
        return forward

    def set_trajectory(self, trajectory):
        self.goal = trajectory[-1]
        self.N = trajectory.shape[0]
        self.direction_changes = self.detect_direction_change(trajectory)
        self.direction_changes.append(self.N - 1)
        self.change = self.direction_changes[0]
        self.dir_idx = 0
        start = 0
        self.trajectory = trajectory

    def detect_direction_change(self, waypoints):
        direction_changes = []
        vectors = np.diff(waypoints, axis=0)
        return direction_changes

    def get_reference_index(self, pose):
        pose = np.array(pose)
        diff = self.trajectory[:, :2] - pose[:2]
        dist = np.linalg.norm(diff[:, :2], axis=1)
        index = dist.argmin()
        self.change_dir = False
        if dist[index] > 0.5:
            self.replan = True
        while (dist[index] < self.waypoint_lookahead and index <= len(self.trajectory) - 2 and index <= self.change):
            index += 1
            index = min(index, len(self.trajectory) - 1)
        if (len(self.trajectory) == 1):
            self.index = 0
            return 0
        if dist[self.N - 1] <= self.threshold and self.dir_idx <= len(
                self.direction_changes) - 2 and index >= self.N - 1:
            # ((pose[:, 2] % (2 * np.pi)) - (self.trajectory[self.change, 2] % (2 * np.pi)))
            self.change_dir = True
            self.dir_idx += 1
            self.trajectory = self.trajectory_list[self.dir_idx]
            self.N = self.trajectory.shape[0]
            index = 0
        if self.dir_idx == (len(self.direction_changes) - 1) and np.linalg.norm(self.goal[:2] - pose[:2]) < 0.15:
            self.sample_null_ = True
        self.index = index
        return index

    def sample_null(self):
        return self.sample_null_

    def tan_dist(self, poses, trajectory):
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().detach().numpy()
        N = poses.shape[0]
        M = trajectory.shape[0]
        A = trajectory[:-1, :2]  # (M-1) x 2
        B = trajectory[1:, :2]  # (M-1) x 2
        AB = B - A  # (M-1) x 2
        AB_norm_sq = np.sum(AB ** 2, axis=1)  # (M-1)
        # Handle zero-length segments
        zero_length = AB_norm_sq == 0
        distances = np.full(N, np.inf)
        if np.any(zero_length):
            A_zero = A[zero_length]  # K x 2
            diff = poses[:, np.newaxis, :2] - A_zero[np.newaxis, :, :]  # N x K x 2
            dist_sq = np.sum(diff ** 2, axis=2)  # N x K
            dist_zero = np.sqrt(dist_sq)  # N x K
            distances = np.minimum(distances, np.min(dist_zero, axis=1))
        # Handle non-zero-length segments
        if np.any(~zero_length):
            A_nonzero = A[~zero_length]  # S x 2
            AB_nonzero = AB[~zero_length]  # S x 2
            AB_norm_sq_nonzero = AB_norm_sq[~zero_length]  # S
            AP = poses[:, np.newaxis, :2] - A_nonzero[np.newaxis, :, :]  # N x S x 2
            numerator = np.einsum('nsi,si->ns', AP, AB_nonzero)  # N x S
            denominator = AB_norm_sq_nonzero  # S
            t = numerator / denominator[np.newaxis, :]  # N x S
            t = np.clip(t, 0, 1)  # Clamp t to [0, 1]
            C = A_nonzero[np.newaxis, :, :] + t[:, :, np.newaxis] * AB_nonzero[np.newaxis, :, :]  # N x S x 2
            diff = poses[:, np.newaxis, :2] - C  # N x S x 2
            dist_sq = np.sum(diff ** 2, axis=2)  # N x S
            dist = np.sqrt(dist_sq)  # N x S
            distances = np.minimum(distances, np.min(dist, axis=1))
        return distances

    def running_cost(self, states, actions):
        car = states[:, :3]
        cost = np.zeros(states.shape[0])

        # Calculate angle difference with proper wrapping
        angle_diff = car[:, 2] - self.trajectory[self.index, 2]
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi

        # Target cost - tracking current trajectory point
        position_cost_x = 1 * (car[:, 0] - self.trajectory[self.index, 0]) ** 2
        position_cost_y = 1 * (car[:, 1] - self.trajectory[self.index, 1]) ** 2
        heading_cost = 2 * (angle_diff) ** 2

        target_cost = position_cost_x + position_cost_y + heading_cost

        # Additional heading velocity cost for better turning anticipation
        if self.index < len(self.trajectory) - 1:
            target_angular_vel = self.trajectory[self.index + 1, 2] - self.trajectory[self.index, 2]
            # Wrap the target angular velocity difference
            target_angular_vel = ((target_angular_vel + np.pi) % (2 * np.pi)) - np.pi

            # Scale heading error by how much turning is required
            heading_velocity_cost = 5 * np.abs(target_angular_vel) * (angle_diff) ** 2
            target_cost += heading_velocity_cost

        # Trajectory following cost
        traj_cost = 1.5 * self.tan_dist(car[:, :2], self.trajectory[:, :2]) ** 2

        # Action cost
        action_cost_throttle = 0.001 * actions[:, 0] ** 2
        action_cost_steering = 0.00001 * actions[:, 1] ** 2
        action_cost = action_cost_throttle + action_cost_steering

        # Combine all costs
        cost = target_cost + traj_cost + action_cost

        return torch.tensor(cost, dtype=torch.float32)

    def terminal_state_cost(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Input:
        s: car state (shape: 1 x NUM_SAMPLES x HORIZON x 3) - [x, y, heading]
        a: car action (shape: 1 x NUM_SAMPLES x HORIZON x 2)
        Output:
        cost associated with each state-action pair (shape: NUM_SAMPLES)
        """
        assert s.ndim == 4 and s.shape[0] == 1 and s.shape[-1] == 3
        assert a.ndim == 4 and a.shape[0] == 1 and a.shape[-1] == 2
        final_states = s[0, :, -1, :]  # shape: NUM_SAMPLES x 3

        # Goal is the last point of the trajectory
        goal_position = self.trajectory[-1, :2]  # [x, y] of final trajectory point
        goal_heading = self.trajectory[-1, 2]  # heading of final trajectory point
        position_cost = torch.sum((final_states[:, :2] - torch.tensor(goal_position, dtype=torch.float32)) ** 2, dim=1)
        angle_diff = final_states[:, 2] - goal_heading
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        heading_cost = 1.5 * (angle_diff) ** 2
        terminal_cost = position_cost + heading_cost
        return terminal_cost

    def car_dynamics(self, states, actions):
        car_states = states
        x_now = car_states[:, 0]
        y_now = car_states[:, 1]
        Th_now = car_states[:, 2]
        dt = 0.01
        speed = actions[:, 1]
        steering_angle = actions[:, 0]
        x_dot = speed * torch.cos(states[:, 2]) * dt
        y_dot = speed * torch.sin(states[:, 2]) * dt
        theta_dot = ((speed * torch.tan(steering_angle)) / (0.295)) * dt
        x_next = x_now + x_dot
        y_next = y_now + y_dot
        Th_next = Th_now + theta_dot
        car_states_next = torch.stack((x_next, y_next, Th_next), dim=1)
        return car_states_next