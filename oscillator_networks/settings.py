# DEFAULTS: Integrating the ODE trajectory
DYNAMICS_METHOD = 'libcall'
DYNAMICS_METHODS_VALID = ['libcall', 'rk4', 'euler']
INIT_COND = [20, 20, 0.0]  # default initial cond for trajectory
NUM_TRAJ = 200  # number of trajectories for trajectory_multiple
TIME_START = 0.0  # standard trajectory start time
TIME_END = 120.0  # standard trajectory end time
NUM_STEPS = 20000  # number of time steps in each trajectory (non-gillespie)
