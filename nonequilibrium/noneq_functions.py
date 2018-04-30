import numpy as np
from random import random

from noneq_constants import BETA, EXT_FIELD_STRENGTH, APP_FIELD_STRENGTH
from noneq_simsetup import N, XI, A_INV, J, CELLTYPE_LABELS, GENE_ID, ETA


def internal_field(state, spin_idx, t):
    internal_field = np.dot(J[spin_idx,:], state[:,t])
    return internal_field


def glauber_dynamics_update(state, spin_idx, t, ext_field=None, ext_field_strength=EXT_FIELD_STRENGTH, app_field=None, app_field_strength=APP_FIELD_STRENGTH):
    """
    See page 107-111 Amit for discussion on functional form
    ext_field - N x 1 - field external to the cell in a signalling sense; exosome field in multicell sym
    ext_field_strength  - scaling factor for ext_field
    app_field - N x 1 - unnatural external field (e.g. force TF on for some time period experimentally)
    app_field_strength - scaling factor for appt_field
    """
    r1 = random()
    total_field = internal_field(state, spin_idx, t)
    if ext_field is not None:
        total_field += ext_field_strength * ext_field[spin_idx]
    if app_field is not None:
        total_field += app_field_strength * app_field[spin_idx]
    prob_on_after_timestep = 1 / (1 + np.exp(-2*BETA*total_field))  # probability that site i will be "up" after the timestep
    if prob_on_after_timestep > r1:
        state[spin_idx, t + 1] = 1.0
    else:
        state[spin_idx, t + 1] = -1.0
    return state

