import matplotlib.pyplot as plt
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count

from singlecell_class import Cell
from singlecell_constants import RUNS_FOLDER, IPSC_CORE_GENES, BETA
from singlecell_data_io import run_subdir_setup
from singlecell_simsetup import N, P, XI, CELLTYPE_ID, A_INV, J, GENE_ID, GENE_LABELS, CELLTYPE_LABELS


ANALYSIS_SUBDIR = "basin_transitions"
ANNEAL_BETA = 1.3
OCC_THRESHOLD = 0.7

def wrapper_get_basin_stats(fn_args_dict):
    np.random.seed()
    if fn_args_dict['kwargs'] is not None:
        return get_basin_stats(*fn_args_dict['args'], **fn_args_dict['kwargs'])
    else:
        return get_basin_stats(*fn_args_dict['args'])


def get_basin_stats(init_cond, init_state, init_id, ensemble, ensemble_idx, num_steps=100, beta=ANNEAL_BETA, anneal=True,
                    verbose=False, occ_threshold=OCC_THRESHOLD):

    # prep applied field TODO: how to include applied field neatly
    # app_field = construct_app_field_from_genes(IPSC_CORE_GENES, num_steps)
    app_field = None

    endpoint_dict = {}
    transfer_dict = {}
    proj_timeseries_array = np.zeros((len(CELLTYPE_LABELS), num_steps))
    basin_occupancy_timeseries = np.zeros((len(CELLTYPE_LABELS) + 1, num_steps), dtype=int)  # could have some spurious here too? not just last as mixed
    mixed_index = len(CELLTYPE_LABELS)  # i.e. last elem

    if anneal:
        beta_start = beta
        beta_end = 2.0
        beta_step = 0.1
        wandering = False

    for cell_idx in xrange(ensemble_idx, ensemble_idx + ensemble):
        print "Simulating cell:", cell_idx
        cell = Cell(init_state, init_id)

        if anneal:
            beta = beta_start  # reset beta to use in each trajectory

        for step in xrange(num_steps):

            # report on each mem proj ranked
            projvec = cell.get_memories_projection()
            proj_timeseries_array[:, step] += projvec
            absprojvec = np.abs(projvec)
            topranked = np.argmax(absprojvec)
            if verbose:
                print "\ncell %d step %d" % (cell_idx, step)

            # print some timestep proj ranking info
            if verbose:
                for rank in xrange(10):
                    sortedmems_smalltobig = np.argsort(absprojvec)
                    sortedmems_bigtosmall = sortedmems_smalltobig[::-1]
                    topranked = sortedmems_bigtosmall[0]
                    ranked_mem_idx = sortedmems_bigtosmall[rank]
                    ranked_mem = CELLTYPE_LABELS[ranked_mem_idx]
                    print rank, ranked_mem_idx, ranked_mem, projvec[ranked_mem_idx], absprojvec[ranked_mem_idx]

            if projvec[topranked] > occ_threshold:
                basin_occupancy_timeseries[topranked, step] += 1
            else:
                basin_occupancy_timeseries[mixed_index, step] += 1

            if topranked != CELLTYPE_ID[init_cond] and projvec[topranked] > occ_threshold:
                if cell_idx not in transfer_dict:
                    transfer_dict[cell_idx] = {topranked: (step, projvec[topranked])}
                else:
                    if topranked not in transfer_dict[cell_idx]:
                        transfer_dict[cell_idx] = {topranked: (step, projvec[topranked])}

            # annealing block
            if projvec[CELLTYPE_ID[init_cond]] < 0.6:
                if verbose:
                    print "+++++++++++++++++++++++++++++++++ Wandering condition passed at step %d" % step
                wandering = True
            elif wandering:
                if verbose:
                    print "+++++++++++++++++++++++++++++++++ Re-entered orig basin after wandering at step %d" % step
                wandering = False
                beta = beta_start
            if anneal and wandering and beta < beta_end:
                beta = beta + beta_step

            # main call to update
            if step < num_steps:
                cell.update_state(beta=beta, app_field=None, fullstep_chunk=True)

        # update endpoints for each cell
        if projvec[topranked] > 0.7:
            endpoint_dict[cell_idx] = (CELLTYPE_LABELS[topranked], projvec[topranked])
        else:
            endpoint_dict[cell_idx] = ('mixed', projvec[topranked])

    return endpoint_dict, transfer_dict, proj_timeseries_array, basin_occupancy_timeseries


def fast_basin_stats(init_cond, init_state, init_id, ensemble, num_processes, num_steps=100, beta=ANNEAL_BETA, anneal=True,
                     verbose=False, occ_threshold=0.7):
    # prepare fn args and kwargs for wrapper
    kwargs_dict = {'num_steps': num_steps, 'beta': beta, 'anneal': anneal, 'verbose': verbose, 'occ_threshold': occ_threshold}
    fn_args_dict = [0]*num_processes
    print "NUM_PROCESSES:", num_processes
    assert ensemble % num_processes == 0
    for i in xrange(num_processes):
        subensemble = ensemble / num_processes
        cell_startidx = i * subensemble
        print "process:", i, "job size:", subensemble, "runs"
        fn_args_dict[i] = {'args': (init_cond, init_state, init_id, subensemble, cell_startidx),
                           'kwargs': kwargs_dict}
    # generate results list over workers
    t0 = time.time()
    pool = Pool(num_processes)
    results = pool.map(wrapper_get_basin_stats, fn_args_dict)
    pool.close()
    pool.join()
    print "TIMER:", time.time() - t0
    # collect pooled results
    summed_endpoint_dict = {}
    summed_transfer_dict = {}
    summed_proj_timeseries_array = np.zeros((len(CELLTYPE_LABELS), num_steps))
    summed_basin_occupancy_timeseries = np.zeros((len(CELLTYPE_LABELS) + 1, num_steps), dtype=int)  # could have some spurious here too? not just last as mixed
    for i, result in enumerate(results):
        endpoint_dict, transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = result
        summed_endpoint_dict.update(endpoint_dict)  # TODO check
        summed_transfer_dict.update(transfer_dict)  # TODO check
        summed_proj_timeseries_array += proj_timeseries_array
        summed_basin_occupancy_timeseries += basin_occupancy_timeseries
    check2 = np.sum(summed_basin_occupancy_timeseries, axis=0)
    print "check2", num_steps, len(check2)
    print check2

    return summed_endpoint_dict, summed_transfer_dict, summed_proj_timeseries_array, summed_basin_occupancy_timeseries


def get_init_info(init_cond):
    """
    Args:
    - init_cond: np array of init state OR string memory label
    Return:
    - init state (Nx1 array) and init_id (str)
    """
    if isinstance(init_cond, np.ndarray):
        init_state = init_cond
        init_id = 'specific'
    else:
        assert isinstance(init_cond, str)
        init_state = XI[:, CELLTYPE_ID[init_cond]]
        init_id = init_cond
    return init_state, init_id


def ensemble_projection_timeseries(init_cond, ensemble, num_processes, num_steps=100, beta=ANNEAL_BETA, anneal=True,
                                   occ_threshold=0.7, plot=True):
    """
    Args:
    - init_cond: np array of init state OR string memory label
    - ensemble: ensemble of particles beginning at init_cond
    - num_steps: how many steps to iterate (each step updates every spin once)
    - beta: simulation temperature parameter
    - occ_threshold: projection value cutoff to say state is in a basin (default: 0.7)
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
    init_state, init_id = get_init_info(init_cond)

    # simulate ensemble - pooled wrapper call
    endpoint_dict, transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = \
        fast_basin_stats(init_cond, init_state, init_id, ensemble, num_processes,
                         num_steps=num_steps, beta=beta, anneal=True, verbose=False, occ_threshold=occ_threshold)

    # normalize proj timeseries
    proj_timeseries_array = proj_timeseries_array / ensemble  # want ensemble average

    # save transition array and run info to file
    proj_timeseries_data = data_folder + os.sep + 'proj_proj_timeseries.txt'
    np.savetxt(proj_timeseries_data, proj_timeseries_array, delimiter=',')
    basin_occupancy_timeseries_data = data_folder + os.sep + 'proj_occupancy_timeseries.txt'
    np.savetxt(basin_occupancy_timeseries_data, basin_occupancy_timeseries, delimiter=',', fmt='%i')

    # plot output
    for idx in xrange(ensemble):
        if idx in transfer_dict:
            print idx, transfer_dict[idx], [CELLTYPE_LABELS[a] for a in transfer_dict[idx].keys()], endpoint_dict[idx]
        else:
            print idx, "no transfer dict entry", endpoint_dict[idx]
    if plot:
        highlights_CLPside = {6:'k', 8: 'blue', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
        savepath_proj = plot_data_folder + os.sep + 'proj_proj_timeseries.png'
        plot_proj_timeseries(proj_timeseries_array, num_steps, ensemble, savepath_proj, highlights=highlights_CLPside)
        savepath_occ = plot_data_folder + os.sep + 'proj_occupancy_timeseries.png'
        plot_basin_occupancy_timeseries(basin_occupancy_timeseries, num_steps, ensemble, occ_threshold, savepath_occ, highlights=highlights_CLPside)
        savepath_endpt = plot_data_folder + os.sep + 'endpt_stats.png'
        plot_basin_endpoints(endpoint_dict, num_steps, ensemble, savepath_endpt, highlights=highlights_CLPside)
    return proj_timeseries_array, basin_occupancy_timeseries


def plot_proj_timeseries(proj_timeseries_array, num_steps, ensemble, savepath, highlights=None):
    """
    proj_timeseries_array is expected dim p x time
    highlights: either None or a dict of idx: color for certain memory projections to highlight
    """
    assert proj_timeseries_array.shape[0] == len(CELLTYPE_LABELS)
    plt.clf()
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


def plot_basin_occupancy_timeseries(basin_occupancy_timeseries, num_steps, ensemble, threshold, savepath, highlights=None):
    """
    basin_occupancy_timeseries: is expected dim (p + spurious tracked) x time  note spurious tracked default is 'mixed'
    highlights: either None or a dict of idx: color for certain memory projections to highlight
    """
    assert basin_occupancy_timeseries.shape[0] == len(CELLTYPE_LABELS) + 1  # note spurious tracked default is 'mixed'
    plt.clf()
    if highlights is None:
        plt.plot(xrange(num_steps), basin_occupancy_timeseries.T, color='blue', linewidth=0.75)
    else:
        plt.plot(xrange(num_steps), basin_occupancy_timeseries.T, color='grey', linewidth=0.55, linestyle='dashed')
        for key in highlights.keys():
            plt.plot(xrange(num_steps), basin_occupancy_timeseries[key,:], color=highlights[key], linewidth=0.75, label=CELLTYPE_LABELS[key])
        if len(CELLTYPE_LABELS) not in highlights.keys():
            plt.plot(xrange(num_steps), basin_occupancy_timeseries[len(CELLTYPE_LABELS), :], color='orange',
                     linewidth=0.75, label='mixed')
        plt.legend()
    plt.title('Occupancy timeseries (ensemble %d)' % ensemble)
    plt.ylabel('Occupancy in each memory (threshold proj=%.2f)' % threshold)
    plt.xlabel('Steps (%d updates, all spins)' % num_steps)
    plt.savefig(savepath)
    return


def plot_basin_endpoints(endpoint_dict, num_steps, ensemble, savepath, highlights=None):
    """
    endpoint_dict: dict where cell endstates stored via endpoint_dict[idx] = (endpoint_label, projval)
    highlights: either None or a dict of idx: color for certain memory projections to highlight
    """

    # TODO: remove endpoint object maybe and just pass basin occupancy timeseries with a step to this fn?

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

        # TODO: store run settings

        # simple analysis
        # common: 'HSC'
        # common: 'Common Lymphoid Progenitor (CLP)'
        # common: 'Common Myeloid Progenitor (CMP)'
        # common: 'Megakaryocyte-Erythroid Progenitor (MEP)'
        # common: 'Granulocyte-Monocyte Progenitor (GMP)'
        # common: 'thymocyte DN'
        # common: 'thymocyte - DP'
        # common: 'neutrophils'
        # common: 'monocytes - classical'
        init_cond = 'HSC'  # note HSC index is 6
        ensemble = 100
        num_steps = 500
        num_proc = cpu_count() / 2  # seems best to use only physical core count (1 core ~ 3x slower than 4)
        ensemble_projection_timeseries(init_cond, ensemble, num_proc, num_steps=num_steps, beta=ANNEAL_BETA, occ_threshold=0.7,
                                       plot=True, anneal=True)
        # less simple analysis
        #basin_transitions()

    # direct data plotting
    if plot_isolated_data:
        loaddata = np.loadtxt(RUNS_FOLDER +os.sep + 'proj_timeseries.txt', delimiter=',')
        ensemble = 100
        highlights_simple = {6:'k', 8: 'blue', 10: 'steelblue'}
        highlights_CLPside = {6:'k', 8: 'blue', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
        highlights_both = {6:'k', 8: 'blue', 10: 'steelblue', 9: 'forestgreen', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
        plot_proj_timeseries(loaddata, loaddata.shape[1], ensemble, RUNS_FOLDER + os.sep + 'proj_timeseries.png',
                             highlights=highlights_CLPside)
