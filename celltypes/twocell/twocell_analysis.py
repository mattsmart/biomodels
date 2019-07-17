import singlecell.init_multiprocessing  # BEFORE numpy
import matplotlib.pyplot as plt
import numpy as np

from singlecell.singlecell_constants import MEMS_MEHTA, MEMS_UNFOLD, BETA, DISTINCT_COLOURS
from singlecell.singlecell_functions import hamiltonian, sorted_energies, label_to_state, get_all_fp, calc_state_dist_to_local_min, partition_basins, reduce_hypercube_dim, state_to_label
from singlecell.singlecell_simsetup import singlecell_simsetup # N, P, XI, CELLTYPE_ID, CELLTYPE_LABELS, GENE_ID
from singlecell.singlecell_visualize import plot_state_prob_map, hypercube_visualize


if __name__ == '__main__':
    # TODO move to singlecell_landscape.py?
