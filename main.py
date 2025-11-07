import gymnasium as gym
from mushr_mujoco_gym.envs.block import MushrBlockEnv
import numpy as np
from scipy.spatial.transform import Rotation as R
from path_tracking_controller import PushingController, SoloCarController
import time
import torch

def pose_quat2euler(pose):
    return np.array(
        [pose[0], pose[1], (np.pi - R.from_quat(pose[2:6]).as_euler('xyz', degrees=False)[0]) % (2 * np.pi)])

def pose_euler2quat(pose):
    quat = R.from_euler('xyz', [0, 0, pose[2]], degrees=False).as_quat()
    return np.array([pose[0], pose[1], quat[3], quat[0], quat[1], quat[2]])


def plot_trajectory(traj, title='Trajectory'):
    """Plot a trajectory with robot orientation arrows"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))

    # Plot path
    plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Path')

    # Plot start and end
    plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
    plt.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=10, label='End')

    # Plot orientation arrows at intervals
    arrow_interval = max(1, len(traj) // 10)
    for i in range(0, len(traj), arrow_interval):
        x, y, theta = traj[i]
        dx = 0.2 * np.cos(theta)
        dy = 0.2 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1,
                  fc='red', ec='red', alpha=0.6)

    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

class Rate:
    def __init__(self, frequency):
        self.period = 1.0 / frequency
        self.last_time = time.time()

    def sleep(self):
        now = time.time()
        elapsed = now - self.last_time
        sleep_time = self.period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_time = time.time()

def generate_curved_path(start_pose, dis=1, curvature=0, num_points=100):
    # curvature = 0, straight line
    x0, y0, theta0 = start_pose
    if curvature == 0 or abs(curvature) < 1e-3:
        s = np.linspace(0, dis, num_points)  # arc length parameter

        x = x0 + s * np.cos(theta0)
        y = y0 + s * np.sin(theta0)
        theta = np.full(num_points, theta0)  # constant heading

        traj = np.stack([x, y, theta], axis=1)
        return traj

    else:
        R = 1.0 / curvature  # turning radius
        delta_theta = dis * curvature
        center_x = x0 - R * np.sin(theta0)
        center_y = y0 + R * np.cos(theta0)
        angle_start = np.arctan2(y0 - center_y, x0 - center_x)
        angles = np.linspace(angle_start, angle_start + delta_theta, num_points)

        x = center_x + R * np.cos(angles)
        y = center_y + R * np.sin(angles)
        theta = angles + np.pi / 2
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        traj = np.stack([x, y, theta], axis=1)
        return traj

def main():
    sim_env = gym.make("MushrBlock-v0", render_mode="human", xml_file="sysid_env3.xml")
    car_start = [1, -1.0, np.pi/2]

    # TODO: bug - block orientation not affected by init state
    block_start = [0.9, -0.7, np.pi/2]
    sim_env.reset()
    init_state = np.concatenate((pose_euler2quat(car_start),
                                 pose_euler2quat(block_start)))
    obs = sim_env.unwrapped.set_init_states(init_state)

    dt = 0.01
    rate = Rate(1/dt)

    start_time = time.time()
    max_time = 200
    push_controller = PushingController()
    straight_line_path = generate_curved_path(block_start, dis=2, curvature=0, num_points=20)
    goal_reached = False
    semi_circle_path = generate_curved_path(block_start, dis=np.pi, curvature=0.5, num_points=50)
    push_controller.set_trajectory(torch.tensor(semi_circle_path, dtype=torch.float32))
    plot_trajectory(semi_circle_path)
    for i in range(100):
        obs, _, _, _, _ = sim_env.step([0, 0])
        rate.sleep()
    while time.time() - start_time < max_time and not goal_reached:
        car_euler = pose_quat2euler(obs[0:6])
        block_euler = pose_quat2euler(obs[6:12])

        # TODO: block orientation bug
        block_euler[2] = block_euler[2] - np.pi/2

        obs_euler = np.concatenate((car_euler, block_euler))

        ref_idx = push_controller.get_reference_index(obs_euler)
        print("error in ref", block_euler-semi_circle_path[ref_idx], "\nref idx", ref_idx)
        if ref_idx == len(semi_circle_path)-1:
            print("reached goal!")
            print("error in goal", block_euler - semi_circle_path[-1])
            goal_reached = True
        action = push_controller.ctrl.command(obs)
        obs, _, _, _, _ = sim_env.step(action)
        rate.sleep()
    sim_env.close()

if __name__ == "__main__":
    main()