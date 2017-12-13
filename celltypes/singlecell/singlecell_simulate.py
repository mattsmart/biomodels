import numpy as np
from random import shuffle

from singlecell_constants import NUM_STEPS
from singlecell_data_io import run_subdir_setup, state_write
from singlecell_functions import glauber_dynamics_update, state_memory_projection, state_memory_overlap, hamiltonian
from singlecell_simsetup import GENE_LABELS, CELLTYPE_LABELS, N, P, XI, A, A_INV, J, ETA, CELLTYPE_ID
from singlecell_visualize import plot_as_radar, save_manual

"""
NOTES:
- projection method seems to be behaving correctly
- TODO: test vs Fig 1E Mehta 2014
- in hopfield sim, at normal temps it jumps immediately to much more stable state and stays there
"""

# IO setup
current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup()

# Variable setup
init_state = XI[:,CELLTYPE_ID['B Cell']] #-1 + np.zeros((N,1))
randomized_sites = range(N)
state = np.zeros((N,NUM_STEPS))
steps = range(NUM_STEPS)
state[:,0] = init_state[:]

# Simulate
for step in steps[:-1]:
    print "step:", step, " H(state) =", hamiltonian(state[:,step])

    if step % 1 == 0:
        #state_vec_proj = state_memory_overlap(state, step)
        state_vec_proj = state_memory_projection(state, step)
        #print state_vec_proj, np.shape(state_vec_proj)
        fig, ax = plot_as_radar(state_vec_proj)
        save_manual(fig, plot_lattice_folder, "sc_state_radar_%d" % step)

    shuffle(randomized_sites)  # randomize site ordering each timestep updates
    for idx, site in enumerate(randomized_sites):  # TODO: parallelize
        state = glauber_dynamics_update(state, idx, step)

# Write
print "Writing state to file.."
print state
state_write(state, steps, GENE_LABELS, "sc_state", "times", "gene_labels", data_folder)
print "Done"
