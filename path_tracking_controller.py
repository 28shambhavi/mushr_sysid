from pytorch_mppi.mppi import MPPI
from torch.distributions.multivariate_normal import MultivariateNormal
from car_cost_function import CarCostFunctions
from pushing_cost_function import PushingCostFunctions
import numpy as np
import torch

class MPPI_R(MPPI):
    def change_direction(self):
        current_max_velocity = self.u_max[1]
        current_min_velocity = self.u_min[1]
        if current_max_velocity > 0:
            new_max_velocity = 0
            new_min_velocity = -current_max_velocity
            mu = -current_max_velocity / 2
        else:
            new_max_velocity = -current_min_velocity
            new_min_velocity = 0
            mu = -current_min_velocity / 2
        self.u_max[1] = new_max_velocity
        self.u_min[1] = new_min_velocity
        self.noise_mu[1] = mu
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        self.reset()

    def set_forward(self):
        current_min_velocity = self.u_min[1]
        if current_min_velocity < 0:
            self.change_direction()

    def set_reverse(self):
        current_max_velocity = self.u_max[1]
        if current_max_velocity > 0:
            self.change_direction()


class SoloCarController():
    def __init__(self):
        self.reset()

    def reset(self):
        self.cost = CarCostFunctions()
        TIMESTEPS = 20  # T
        N_SAMPLES = 100  # K
        ACTION_LOW = [-0.34, 0]
        ACTION_HIGH = [0.34, 0.4]

        self.d = 'cpu'
        nx = 3
        noise_sigma = torch.tensor([[0.03, 0.0],
                                    [0.0, .05]], device=self.d, dtype=torch.float32)
        lambda_ = 1e-2
        noise_mu = torch.tensor([0.0, 0.2], device=self.d, dtype=torch.torch.float32)

        self.ctrl = MPPI_R(self.cost.car_dynamics, self.cost.running_cost, nx=nx, noise_sigma=noise_sigma,
                           num_samples=N_SAMPLES, horizon=TIMESTEPS,
                           lambda_=lambda_, device=self.d, noise_mu=noise_mu,
                           u_min=torch.tensor(ACTION_LOW, dtype=torch.float32, device=self.d),
                           u_max=torch.tensor(ACTION_HIGH, dtype=torch.float32, device=self.d),
                           sample_null_action=False)

        print("Initalized MPPI Controller")

        self.index = 0
        self.forward = True
        self.reached_goal = False
        self.replan = False

    def set_trajectory(self, trajectory):
        self.reset()
        self.cost.set_trajectory(trajectory)
        self.replan = False
        if self.cost.start_forward == False:
            self.ctrl.set_reverse()

    def get_reference_index(self, obs):
        self.index = self.cost.get_reference_index(obs)
        if self.cost.change_dir == True:
            self.ctrl.change_direction()
        self.ctrl.sample_null_action = self.cost.sample_null()

        self.reached_goal = self.cost.sample_null()
        self.replan = self.cost.replan

        return self.index

    def check_direction(self, car, ref):
        v = np.array([ref[0] - car[0], ref[1] - car[1]])
        d = np.array([np.cos(car[2]), np.sin(car[2])])

        dp = np.dot(v, d)
        if dp >= 0:
            forward = True
        else:
            forward = False

        if forward != self.forward:
            if forward == True:
                self.ctrl.set_forward()
            else:
                self.ctrl.set_reverse()

        self.forward = forward


class PushingController():
    def __init__(self):
        self.reset()

    def reset(self):
        self.controller_active = False
        self.cost = PushingCostFunctions(device=torch.device("cpu"))

        TIMESTEPS = 30  # T
        N_SAMPLES = 200  # K
        ACTION_LOW = [-0.17, 0]
        ACTION_HIGH = [0.17, 0.21]

        d = torch.device("cpu")
        nx = 6  # state dimension
        noise_sigma = torch.tensor([[0.05, 0.0],
                                    [0.0, .09]], device=d, dtype=torch.float32)
        lambda_ = 1e-2
        noise_mu = torch.tensor([0.0, 0.0], device=d, dtype=torch.torch.float32)

        self.ctrl = MPPI(self.cost.push_dynamics, self.cost.running_cost, nx=nx, noise_sigma=noise_sigma,
                         num_samples=N_SAMPLES, horizon=TIMESTEPS,
                         lambda_=lambda_, device=d, terminal_state_cost=self.cost.terminal_state_cost, noise_mu=noise_mu,
                         u_min=torch.tensor(ACTION_LOW, dtype=torch.torch.float32, device=d),
                         u_max=torch.tensor(ACTION_HIGH, dtype=torch.torch.float32, device=d), noise_abs_cost=False,
                         sample_null_action=False)

        self.index = 0
        self.reached_goal = False

    def set_trajectory(self, trajectory):
        self.reset()
        self.cost.set_trajectory(trajectory)

    def get_reference_index(self, obs):
        self.index = self.cost.get_reference_index(obs)
        self.ctrl.sample_null_action = self.cost.sample_null()
        self.reached_goal = self.cost.sample_null()
        return self.index
