import numpy as np
from random import shuffle

from singlecell_constants import NUM_STEPS
from singlecell_data_io import run_subdir_setup, state_write
from singlecell_functions import glauber_dynamics_update, state_memory_projection, state_memory_overlap
from singlecell_simsetup import GENE_LABELS, CELLTYPE_LABELS, N, P, XI, A, A_INV, J, ETA, CELLTYPE_ID
from singlecell_visualize import plot_as_radar, save_manual


# IO setup
current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup()

# Variable setup
init_state = XI[:,12] #-1 + np.zeros((N,1))
print CELLTYPE_ID
print CELLTYPE_ID['B Cell']
print np.shape(XI), np.shape(init_state), A[CELLTYPE_ID['B Cell'], CELLTYPE_ID['T Cell']]
randomized_sites = range(N)
state = np.zeros((N,NUM_STEPS))
steps = range(NUM_STEPS)
state[:,0] = init_state[:]

# Simulate
for step in steps[:-1]:
    shuffle(randomized_sites)  # randomize site ordering each timestep updates
    for idx, site in enumerate(randomized_sites):  # TODO: parallelize
        state = glauber_dynamics_update(state, idx, step)

    #state_vec_proj = state_memory_overlap(state, step)
    state_vec_proj = state_memory_projection(state, step)
    print state_vec_proj, np.shape(state_vec_proj)
    fig, ax = plot_as_radar(state_vec_proj)
    save_manual(fig, plot_lattice_folder, "sc_state_radar_%d" % step)
    print step

# Write
print state
print "Writing state to file.."
state_write(state, steps, GENE_LABELS, "sc_state", "times", "gene_labels", data_folder)
print "Done"
