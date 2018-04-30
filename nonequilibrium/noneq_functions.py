import numpy as np
from random import random

from noneq_constants import BETA


def internal_field(state, spin_idx, t):
    internal_field = np.dot(J[spin_idx,:], state[:,t])
    return internal_field


def glauber_dynamics_update(state, spin_idx, t, app_field=None):
    r1 = random()
    total_field = internal_field(state, spin_idx, t)
    if app_field is not None:
        total_field += app_field[spin_idx]
    prob_on_after_timestep = 1 / (1 + np.exp(-2*BETA*total_field))  # probability that site i will be "up" after the timestep
    if prob_on_after_timestep > r1:
        state[spin_idx, t + 1] = 1.0
    else:
        state[spin_idx, t + 1] = -1.0
    return state
