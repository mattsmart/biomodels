import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell_class import Cell
from singlecell_constants import RUNS_FOLDER, IPSC_CORE_GENES, BETA
from singlecell_data_io import run_subdir_setup
from singlecell_simsetup import N, P, XI, CELLTYPE_ID, A_INV, J, GENE_ID, GENE_LABELS, CELLTYPE_LABELS


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
    if anneal:
        beta_start = beta
        beta_end = 2.0
        beta_step = 0.1
        wandering = False

    # simulate ensemble
    endpoint_dict = {}
    transfer_dict = {}

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
            topranked = sortedmems_bigtosmall[0]
            print "\ncell %d step %d" % (cell_idx, step)

            # print some timestep proj ranking info
            for rank in xrange(10):
                ranked_mem_idx = sortedmems_bigtosmall[rank]
                ranked_mem = CELLTYPE_LABELS[ranked_mem_idx]
                print rank, ranked_mem_idx, ranked_mem, projvec[ranked_mem_idx], absprojvec[ranked_mem_idx]

            if topranked != CELLTYPE_ID[init_cond] and projvec[topranked] > 0.75:
                if cell_idx not in transfer_dict:
                    transfer_dict[cell_idx] = {topranked: (step, projvec[topranked])}
                else:
                    if topranked not in transfer_dict[cell_idx]:
                        transfer_dict[cell_idx] = {topranked: (step, projvec[topranked])}

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
            if step < num_steps:
                cell.update_state(beta=beta, app_field=None)  # TODO alternate update random site at a time scheme

        # update endpoints for each cell
        if topranked > projvec[topranked] > 0.7:
            endpoint_dict[cell_idx] = (CELLTYPE_LABELS[topranked], projvec[topranked])
        else:
            endpoint_dict[cell_idx] = ('mixed', projvec[topranked])

    proj_timeseries_array = proj_timeseries_array / ensemble  # want ensemble average

    # save transition array and run info to file
    proj_timeseries_data = data_folder + os.sep + 'proj_timeseries.txt'
    np.savetxt(proj_timeseries_data, proj_timeseries_array, delimiter=',')

    # plot output
    for idx in xrange(ensemble):
        if idx in transfer_dict:
            print idx, transfer_dict[idx], [CELLTYPE_LABELS[a] for a in transfer_dict[idx].keys()], endpoint_dict[idx]
        else:
            print idx, "no transfer dict entry", endpoint_dict[idx]
    if plot:
        highlights_CLPside = {6:'k', 8: 'blue', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
        savepath = plot_data_folder + os.sep + 'proj_timeseries.pdf'
        plot_proj_timeseries(proj_timeseries_array, num_steps, ensemble, savepath, highlights=highlights_CLPside)

        savepath_endpt = plot_data_folder + os.sep + 'endpt_stats.pdf'
        plot_basin_endpoints(endpoint_dict, num_steps, ensemble, savepath_endpt, highlights=highlights_CLPside)

    return proj_timeseries_array


def plot_proj_timeseries(proj_timeseries_array, num_steps, ensemble, savepath, highlights=None):
    """
    proj_timeseries_array is expected dim p x time
    highlights: either None or a dict of idx: color for certain memory projections to highlight
    """
    assert proj_timeseries_array.shape[0] == len(CELLTYPE_LABELS)
    if highlights is None:
        plt.plot(xrange(num_steps), proj_timeseries_array.T, color='blue', linewidth=0.75)
    else:
        plt.plot(xrange(num_steps), proj_timeseries_array.T, color='grey', linewidth=0.55, linestyle='dashed')
        for key in highlights.keys():
            plt.plot(xrange(num_steps), proj_timeseries_array[key,:], color=highlights[key], linewidth=0.75, label=CELLTYPE_LABELS[key])
        plt.legend()
    plt.title('Ensemble mean (n=%d) projection timeseries' % ensemble)
    plt.ylabel('Mean projection onto each memory')
    plt.xlabel('Steps (%d updates, all spins)' % num_steps)
    plt.savefig(savepath)
    return


def plot_basin_endpoints(endpoint_dict, num_steps, ensemble, savepath, highlights=None):
    """
    endpoint_dict: dict where cell endstates stored via endpoint_dict[idx] = (endpoint_label, projval)
    highlights: either None or a dict of idx: color for certain memory projections to highlight
    """
    # data prep
    occupancies = {}
    for idx in xrange(len(endpoint_dict.keys())):
        endppoint_id, projval = endpoint_dict[idx]
        if endppoint_id in occupancies:
            occupancies[endppoint_id] += 1
        else:
            occupancies[endppoint_id] = 1
    memory_labels = occupancies.keys()
    memory_occupancies = [occupancies[a] for a in memory_labels]
    memory_colors = ['grey' if label not in [CELLTYPE_LABELS[a] for a in highlights.keys()]
                     else highlights[CELLTYPE_ID[label]]
                     for label in memory_labels]
    # plotting
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 12})

    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(18.5, 10.5)
    h = plt.bar(xrange(len(memory_labels)), memory_occupancies, color=memory_colors, label=memory_labels)
    plt.subplots_adjust(bottom=0.3)
    xticks_pos = [0.65 * patch.get_width() + patch.get_xy()[0] for patch in h]
    plt.xticks(xticks_pos, memory_labels, ha='right', rotation=45, size=12)
    plt.gca().yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.title('Cell endpoints (%d steps, %d cells)' % (num_steps, ensemble))
    plt.ylabel('Basin occupancy count')
    plt.xlabel('Basin labels')
    fig.savefig(savepath)
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
    gen_basin_data = True
    plot_isolated_data = False

    if gen_basin_data:
        # simple analysis
        init_cond = 'HSC'  # index is 6
        ensemble = 100
        ensemble_projection_timeseries(init_cond, ensemble, num_steps=25, beta=1.3, plot=True, anneal=True)
        # less simple analysis
        #basin_transitions()

    # direct data plotting
    if plot_isolated_data:
        loaddata = np.loadtxt('proj_timeseries.txt', delimiter=',')
        ensemble = 100
        highlights_simple = {6:'k', 8: 'blue', 10: 'steelblue'}
        highlights_CLPside = {6:'k', 8: 'blue', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
        highlights_both = {6:'k', 8: 'blue', 10: 'steelblue', 9: 'forestgreen', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
        plot_proj_timeseries(loaddata, loaddata.shape[1], ensemble, 'proj_timeseries.pdf', highlights=highlights_CLPside)
