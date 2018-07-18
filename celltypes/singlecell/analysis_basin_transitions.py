"""
Track jumps from basin 'i' to basin 'j' for all 'i'

Defaults:
- temperature: default is intermediate
- ensemble: 10,000 cells start in basin 'i'
- time: fixed, 100 steps (option for unspecified; stop when ensemble dissipates)

Output:
- matrix of basin transition probabilities (i.e. discrete time markov chain)

Spurious basin notes:
- unclear how to identify spurious states dynamically
- suppose
- define new spurious state if, within some window of time T:
    - (A) the state does not project on planned memories within some tolerance; and
    - (B) the state has some self-similarity over time
- if a potential function is known (e.g. energy H(state)) then a spurious state
  could be formally defined as a minimizer; however this may be numerically expensive to check
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell_constants import RUNS_FOLDER, IPSC_CORE_GENES
from singlecell_functions import state_burst_errors, state_memory_projection_single, construct_app_field_from_genes
from singlecell_simsetup import N, P, XI, CELLTYPE_ID, A_INV, J, GENE_ID, GENE_LABELS, CELLTYPE_LABELS
from singlecell_simulate import singlecell_sim


def basin_transitions():
    analysis_subdir = "basin_transitions"
    num_steps = 100
    """
    app_field = construct_app_field_from_genes(IPSC_CORE_GENES, num_steps)
    proj_timeseries_array = np.zeros((num_steps, P))
    """

    # add 1 as spurious sink dimension? this treats spurious as global sink state
    basins_dim = len(CELLTYPE_LABELS) + 1
    spurious_index = len(CELLTYPE_LABELS)
    transition_data = np.zeros((basins_dim,basins_dim))

    for idx, memory_label in enumerate(CELLTYPE_LABELS):
        # TODO
        print idx, memory_label
        """
        cellstate_array, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = singlecell_sim(init_id=memory_label, iterations=num_steps, app_field=app_field, app_field_strength=10.0,
                                                                                                                 flag_burst_error=FLAG_BURST_ERRORS, flag_write=False, analysis_subdir=analysis_subdir,
                                                                                                                 plot_period=num_steps*2)
        proj_timeseries_array[:, idx] = get_memory_proj_timeseries(cellstate_array, esc_idx)[:]
        """
        # TODO: transiton_data_row = ...
        transiton_data_row = 0

        transition_data[idx, :] = transiton_data_row


    # cleanup output folders from main()
    # TODO...

    # save transition array and run info to file
    # TODO...

    # plot output
    # TODO...
    figname = 'basin_transitions.png'
    plt.plot(xrange(num_steps), proj_timeseries_array, color='blue', linewidth=0.75)
    plt.title('Basin Transitions %s' % num_steps)
    plt.ylabel('Basin Transitions %s' % num_steps)
    plt.xlabel('Time (%d updates, all spins)' % num_steps)
    plt.savefig(RUNS_FOLDER + analysis_subdir + os.sep + figname)

    return transition_data


if __name__ == '__main__':
    basin_transitions()
