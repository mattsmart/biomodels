import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell_constants import RUNS_FOLDER, IPSC_CORE_GENES, BETA
from singlecell_data_io import run_subdir_setup
from singlecell_functions import state_burst_errors, state_memory_projection_single, construct_app_field_from_genes, \
                                 state_memory_projection
from singlecell_simsetup import N, P, XI, CELLTYPE_ID, A_INV, J, GENE_ID, GENE_LABELS, CELLTYPE_LABELS
from singlecell_simulate import singlecell_sim
from singlecell_class import Cell


ANALYSIS_SUBDIR = "basin_transitions"


def ensemble_projection_timeseries(init_cond, ensemble, num_steps=100, beta=BETA, plot=True):
    """
    Args:
    - init_cond: np array of init state OR string memory label
    - ensemble: ensemble of particles beginning at init_cond
    - num_steps: how many steps to iterate (each step updates every spin once)
    - beta: simulation temperature parameter
    What:
    - Track a timeseries of: ensemble mean projection onto each memory
    - Optionally plot
    Eeturn:
    - timeseries of projections onto store memories (dim p x T)
    """

    # prep io
    current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = \
        run_subdir_setup(run_subfolder=ANALYSIS_SUBDIR)

    # generate initial state
    if isinstance(init_cond, np.ndarray):
        init_state = init_cond
        init_id = 'specific'
    else:
        assert isinstance(init_cond, str)
        print CELLTYPE_ID
        init_state = XI[:, CELLTYPE_ID[init_cond]]
        init_id = init_cond

    # prep applied field TODO: how to include applied field neatly
    # app_field = construct_app_field_from_genes(IPSC_CORE_GENES, num_steps)
    app_field = None

    # simulate ensemble
    proj_timeseries_array = np.zeros((len(CELLTYPE_LABELS), num_steps))
    for cell_idx in xrange(ensemble):
        print "Simulating cell:", cell_idx
        cell = Cell(init_state, init_id)
        for step in xrange(num_steps):
            proj_timeseries_array[:, step] += cell.get_memories_projection()
            cell.update_state(beta=beta, app_field=None)  # TODO alternate update random site at a time scheme
    proj_timeseries_array = proj_timeseries_array / ensemble  # want ensemble average

    # save transition array and run info to file
    proj_timeseries_data = data_folder + os.sep + 'proj_timeseries.txt'
    np.savetxt(proj_timeseries_data, proj_timeseries_array, delimiter=',')

    # plot output
    if plot:
        proj_timeseries_plot = plot_data_folder + os.sep + 'proj_timeseries.pdf'
        plt.plot(xrange(num_steps), proj_timeseries_array.T, color='blue', linewidth=0.75)
        plt.title('Ensemble mean (n=%d) projection timeseries' % ensemble)
        plt.ylabel('Mean projection onto each memory')
        plt.xlabel('Steps (%d updates, all spins)' % num_steps)
        plt.savefig(proj_timeseries_plot)

    return proj_timeseries_array


def basin_transitions(init_cond, ensemble, num_steps, beta):
    """
    Track jumps from basin 'i' to basin 'j' for all 'i'

    Defaults:
    - temperature: default is intermediate (1/BETA from singlecell_constants)
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

    return transition_data


if __name__ == '__main__':
    # simple analysis
    init_cond = 'HSC'  # index is 6
    ensemble = 100
    ensemble_projection_timeseries(init_cond, ensemble, num_steps=100, beta=0.5, plot=True)
    # less simple analysis
    #basin_transitions()
