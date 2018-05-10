import numpy as np

from cytokine_settings import build_model, DEFAULT_MODEL, APP_FIELD_STRENGTH, RUNS_SUBDIR_CYTOKINES
from cytokine_simulate import cytokine_sim

from singlecell_class import Cell
from singlecell_constants import NUM_STEPS, BETA
from singlecell_data_io import run_subdir_setup
from singlecell_functions import state_to_label, label_to_state


def state_landscape(model_name=DEFAULT_MODEL, iterations=NUM_STEPS, applied_field_strength=APP_FIELD_STRENGTH, flag_write=False):

    spin_labels, intxn_matrix, applied_field_const, init_state = build_model(DEFAULT_MODEL)
    N = len(spin_labels)

    labels_to_states = {idx:label_to_state(idx, N) for idx in xrange(2 ** N)}
    states_to_labels = {tuple(v): k for k, v in labels_to_states.iteritems()}

    for state_label in range(2**N):
        init_cond = labels_to_states[state_label]
        print "\n\nSimulating with init state label", state_label, ":", init_cond
        state_array, dirs = cytokine_sim(iterations=iterations, beta=1000.0, flag_write=False,
                                         applied_field_strength=applied_field_strength, init_state_force=init_cond)
        label_timeseries = [states_to_labels[tuple(state_array[:,t])] for t in xrange(iterations)]
        for elem in label_timeseries:
            print elem, "|",
    return


if __name__ == '__main__':
    # For model A:
    # - deterministic oscillations between state 0 (all-off) and state 15 (all-on)
    # - if sufficient field is added, the oscillations disappear and its just stuck in the all-on state 15
    # - threshold h_0 strength is cancelling the negative feedback term J_2on0 = J[0,2] of SOCS (s_2) on R (s_0)
    state_landscape(iterations=10, applied_field_strength=0.0)
