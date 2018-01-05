import numpy as np

from singlecell_functions import state_burst_errors
from singlecell_simsetup import N, XI, CELLTYPE_ID, A_INV
from singlecell_simulate import main


esc_label = 'esc'
esc_idx = CELLTYPE_ID[esc_label]
init_state_esc = XI[:, esc_idx]


for ratio_to_flip in np.linspace(0.0, 0.5, 50):
    subsample_state = state_burst_errors(init_state_esc, ratio_to_flip=ratio_to_flip)
    cellstate_array = main(init_state=subsample_state, plot_period=100)
    proj_vector = np.dot(A_INV, state_memory_overlap(state_arr