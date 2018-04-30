import numpy as np
from random import shuffle

from noneq_constants import NUM_STEPS, DEFAULT_N
from noneq_data_io import state_write
from noneq_functions import glauber_dynamics_update
from noneq_data_io import run_subdir_setup
from noneq_simulate import state_simulate

# settings
ensemble_size = int(1e6)
total_steps = int(1e6)
N = 3
J = np.array([[0,1,1],
              [1,0,1],
              [1,1,0]])


def get_steadystate_dist(ensemble_size, total_steps, N, J)
    for system in xrange(ensemble_size):
        state_array, _, _, _, _ = state_simulate(init_state=None, init_id=None, N=N, iterations=total_steps, intxn_matrix=J,
                                                 app_field=None, flag_write=False, analysis_subdir="ensemble", plot_period=10)
        state_ending = state_array[-1, :]


state_simulate()