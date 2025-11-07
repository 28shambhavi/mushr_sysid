# mushr_sysid

pushing controller parameters for MPPI
```
TIMESTEPS = 30  # time horizon
N_SAMPLES = 200  # number of samples
ACTION_LOW = [-0.17, 0] # [min steering, min velocity]
ACTION_HIGH = [0.17, 0.21] # [max steering, max velocity]
nx = 6 # state dimension
noise [[0.05, 0.0],
        [0.0, .09]]
lambda_ = 1e-2
noise_mu = [0.0, 0.0]
```

curved path generation
```
curvature = 1/R # 0 is straight line, curvature 1/R forms arc of radius R
dis = 2*pi*R # distance to be traveled in the path 2piR for full circle
num_points = 100 # num of waypoints
```

pushing cost function
```
waypoint_lookahead = 0.16 
threshold = 0.08
sample_null_ = False 
replan = False
terminal_scale = 100.0
```

cost function used for path tracking

MPC cost function class for autonomous vehicle trajectory tracking using a bicycle model.

### Trajectory Setup
- **`set_trajectory(trajectory)`** -
- Sets reference path and goal waypoint
- **`get_reference_index(pose)`** 
- - Finds current target waypoint with 0.08 unit lookahead

### Cost Computation
**`running_cost(states, actions)`** 
- Returns instantaneous cost:
- **Target tracking**: Position (weight=1.0) + Heading (weight=2.0)
- **Turn anticipation**: Heading velocity cost (weight=5.0)
- **Path following**: Distance to trajectory using tangential_distance function (weight=1.5)
- **Control effort**: Throttle (0.001) + Steering (0.00001)
- **Heading difference**: Cost for alignment/orientation error in car and block

**`terminal_state_cost(s, a)`** 
- Final state cost (scaled 100x):
- Position error to goal (weight=1.0)
- Heading error to goal (weight=1.5)

- **`tan_dist(poses, trajectory)`** - Used for calculating perpendicular distance of pose from path segments

### functions
- **`pushing_dynamics(states, actions)`** 
- - Bicycle model (wheelbase=0.295, dt=0.01)
  - State: [x, y, heading, block_x, block_y, block_heading]
  - Action: [steering_angle, speed]
