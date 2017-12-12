import numpy as np
from random import shuffle

from singlecell_constants import BETA, TIMESTEPS
from singlecell_data_io import run_subdir_setup, state_write
from singlecell_functions import memory_corr_matrix, interaction_matrix, local_field
from singlecell_simsetup import GENE_LABELS, CELLTYPE_LABELS, N, P, XI, A_INV, J, ETA


# Variable setup
init_state = -1 + np.zeros((N,1))
randomized_sites = range(N)
state = np.zeros((N,TIMESTEPS))
times = range(TIMESTEPS)
state[:,0] = init_state[:,0]

# Simulate
for t in times[:-1]:
    shuffle(randomized_sites)  # randomize site ordering each timestep updates
    for idx, site in enumerate(randomized_sites):  # TODO: parallelize
        state = glauber_dynamics_update(J, state, idx, t)
    print t

# Write
print state
print "Writing state to file.."
current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup()
state_write(state, times, gene_labels, "sc_state", "times", "gene_labels", data_folder)
print "Done"
