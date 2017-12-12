import numpy as np
from random import shuffle

from singlecell_constants import NUM_STEPS
from singlecell_data_io import run_subdir_setup, state_write
from singlecell_functions import glauber_dynamics_update
from singlecell_simsetup import GENE_LABELS, CELLTYPE_LABELS, N, P, XI, A_INV, J, ETA


# Variable setup
init_state = -1 + np.zeros((N,1))
randomized_sites = range(N)
state = np.zeros((N,NUM_STEPS))
steps = range(NUM_STEPS)
state[:,0] = init_state[:,0]

# Simulate
for step in steps[:-1]:
    shuffle(randomized_sites)  # randomize site ordering each timestep updates
    for idx, site in enumerate(randomized_sites):  # TODO: parallelize
        state = glauber_dynamics_update(state, idx, step)
    print step

# Write
print state
print "Writing state to file.."
current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup()
state_write(state, steps, GENE_LABELS, "sc_state", "times", "gene_labels", data_folder)
print "Done"
