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


def ensemble_projection_timeseries(init_cond, ensemble, num_steps=100, beta=BETA, anneal=True, plot=True):
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
        init_state = XI[:, CELLTYPE_ID[init_cond]]
        init_id = init_cond

    # prep applied field TODO: how to include applied field neatly
    # app_field = construct_app_field_from_genes(IPSC_CORE_GENES, num_steps)
    app_field = None

    # prep temp timeseries (local annealing)
    # TODO: implement annealing -- either pre-plan temperature or use in-loop conditionals
    """
    if np.isscalar(beta):
        print "NOTE: fixed temperature provided -- no annealing"
        beta_series = [beta for _ in xrange(num_steps)]
    else:
        print "NOTE: fixed temperature provided -- no annealing"
        beta_series = [beta for _ in xrange(num_steps)]
    """
    if anneal:
        beta_start = beta
        beta_end = 2.0
        beta_step = 0.1
        wandering = False

    # simulate ensemble
    proj_timeseries_array = np.zeros((len(CELLTYPE_LABELS), num_steps))
    for cell_idx in xrange(ensemble):
        print "Simulating cell:", cell_idx
        cell = Cell(init_state, init_id)

        if anneal:
            beta = beta_start  # reset beta to use in each trajectory

        for step in xrange(num_steps):
            #proj_timeseries_array[:, step] += cell.get_memories_projection()

            # report on each mem proj ranked
            projvec = cell.get_memories_projection()
            proj_timeseries_array[:, step] += projvec
            absprojvec = np.abs(projvec)
            sortedmems_smalltobig = np.argsort(absprojvec)
            sortedmems_bigtosmall = sortedmems_smalltobig[::-1]
            print "\ncell %d step %d" % (cell_idx, step)
            for rank in xrange(10):
                ranked_mem_idx = sortedmems_bigtosmall[rank]
                ranked_mem = CELLTYPE_LABELS[ranked_mem_idx]
                print rank, ranked_mem_idx, ranked_mem, projvec[ranked_mem_idx], absprojvec[ranked_mem_idx]

            # annealing block
            if projvec[CELLTYPE_ID[init_cond]] < 0.6:
                print "+++++++++++++++++++++++++++++++++ Wandering condition passed at step %d" % step
                wandering = True
            elif wandering:
                print "+++++++++++++++++++++++++++++++++ Re-entered orig basin after wandering at step %d" % step
                wandering = False
                beta = beta_start
            if anneal and wandering and beta < beta_end:
                beta = beta + beta_step

            # main call to update
            cell.update_state(beta=beta, app_field=None)  # TODO alternate update random site at a time scheme

    proj_timeseries_array = proj_timeseries_array / ensemble  # want ensemble average

    # save transition array and run info to file
    proj_timeseries_data = data_folder + os.sep + 'proj_timeseries.txt'
    np.savetxt(proj_timeseries_data, proj_timeseries_array, delimiter=',')

    # plot output
    if plot:
        proj_timeseries_plot = plot_data_folder + os.sep + 'proj_timeseries.pdf'
        proj_timeseries_plot(proj_timeseries_array, num_steps, ensemble, proj_timeseries_plot)

    return proj_timeseries_array


def proj_timeseries_plot(proj_timeseries_array, num_steps, ensemble, savepath, highlights=None):
    """
    proj_timeseries_array is expected dim p x time
    highlights: either None or a list of tuples: (idx, color) for certain memory projections to highlight
    """
    assert proj_timeseries_array.shape[0] == len(CELLTYPE_LABELS)
    if highlights is None:
        plt.plot(xrange(num_steps), proj_timeseries_array.T, color='blue', linewidth=0.75)
    else:
        plt.plot(xrange(num_steps), proj_timeseries_array.T, color='grey', linewidth=0.55, linestyle='dashed')
        for pair in highlights:
            plt.plot(xrange(num_steps), proj_timeseries_array[pair[0],:], color=pair[1], linewidth=0.75, label=CELLTYPE_LABELS[pair[0]])
        plt.legend()
    plt.title('Ensemble mean (n=%d) projection timeseries' % ensemble)
    plt.ylabel('Mean projection onto each memory')
    plt.xlabel('Steps (%d updates, all spins)' % num_steps)
    plt.savefig(savepath)
    return


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

    gen_basin_data = False
    plot_isolated_data = True

    if gen_basin_data:
        # simple analysis
        init_cond = 'HSC'  # index is 6
        ensemble = 100
        ensemble_projection_timeseries(init_cond, ensemble, num_steps=100, beta=1.3, plot=True, anneal=True)
        # less simple analysis
        #basin_transitions()

    # direct data plotting
    if plot_isolated_data:
        loaddata = np.loadtxt('proj_timeseries.txt', delimiter=',')
        ensemble = 100
        highlights_simple = [(8, 'blue'), (10, 'steelblue')]
        highlights_CLPside = [(8, 'blue'), (7, 'red'), (16, 'deeppink'), (11, 'darkorchid')]
        highlights_both = [(8, 'blue'), (10, 'steelblue'), (9, 'forestgreen'), (7, 'red'), (16, 'deeppink'), (11, 'darkorchid')]
        proj_timeseries_plot(loaddata, loaddata.shape[1], ensemble, 'proj_timeseries.pdf', highlights=highlights_CLPside)