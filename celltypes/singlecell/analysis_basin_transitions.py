import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count

from analysis_basin_plotting import plot_proj_timeseries, plot_basin_occupancy_timeseries, plot_basin_endpoints
from singlecell_class import Cell
from singlecell_constants import RUNS_FOLDER, IPSC_CORE_GENES, BETA
from singlecell_data_io import run_subdir_setup, runinfo_append
from singlecell_simsetup import N, P, XI, CELLTYPE_ID, A_INV, J, GENE_ID, GENE_LABELS, CELLTYPE_LABELS


ANALYSIS_SUBDIR = "basin_transitions"
ANNEAL_BETA = 1.3
ANNEAL_PROTOCOL = "protocol_A"
FIELD_PROTOCOL = None
OCC_THRESHOLD = 0.7


def field_setup(protocol=FIELD_PROTOCOL):
    """
    """
    # TODO build
    assert protocol in [None]
    field_dict = {'protocol': protocol,
                   'field_start': None}
    return field_dict


def anneal_setup(protocol=ANNEAL_PROTOCOL):
    """
    Start in basin of interest at some intermediate temperature that allows basin escape
    (e.g. beta_init = 1 / T_init = 1.3)
    For each trajectory:
    - Modify temperature once it has left basin
      (define by projection vs a strict cutoff, e.g. if projection[mem_init] < 0.6, then the trajectory is wandering)
    - Modification schedule: Decrease temp each timestep (beta += beta_step) until some ceiling is reached (beta_end)
    - Note currently "one timestep" is N spin flips.
    - If particle re-enters basin (using same cutoff as above), reset beta to beta_init and repeat procedure.
    """
    assert protocol in ["constant", "protocol_A", "protocol_B"]
    anneal_dict = {'protocol': protocol,
                   'beta_start': ANNEAL_BETA}
    if protocol == "protocol_A":
        anneal_dict.update({'beta_end': 2.0,
                            'beta_step': 0.1,
                            'wandering_threshold': 0.6})
    elif protocol == "protocol_B":
        anneal_dict.update({'beta_end': 3.0,
                            'beta_step': 0.5,
                            'wandering_threshold': 0.6})
    else:
        assert protocol == "constant"
        anneal_dict.update({'beta_end': ANNEAL_BETA,
                            'beta_step': 0.0,
                            'wandering_threshold': 0.6})
    return anneal_dict


def anneal_iterate(proj_onto_init, beta_current, step, wandering, anneal_dict, verbose=False):
    if proj_onto_init < anneal_dict['wandering_threshold']:
        if verbose:
            print "+++++++++++++++++++++++++++++++++ Wandering condition passed at step %d" % step
        wandering = True
    elif wandering:
        if verbose:
            print "+++++++++++++++++++++++++++++++++ Re-entered orig basin after wandering at step %d" % step
        wandering = False
        beta_current = anneal_dict['beta_start']
    if wandering and beta_current < anneal_dict['beta_end']:
        beta_current = beta_current + anneal_dict['beta_step']
    return beta_current, wandering


def wrapper_get_basin_stats(fn_args_dict):
    np.random.seed()
    if fn_args_dict['kwargs'] is not None:
        return get_basin_stats(*fn_args_dict['args'], **fn_args_dict['kwargs'])
    else:
        return get_basin_stats(*fn_args_dict['args'])


def get_basin_stats(init_cond, init_state, init_id, ensemble, ensemble_idx, num_steps=100,
                    anneal_protocol=ANNEAL_PROTOCOL, field_protocol=FIELD_PROTOCOL, occ_threshold=OCC_THRESHOLD, verbose=False):

    # prep applied field TODO: how to include applied field neatly
    # app_field = construct_app_field_from_genes(IPSC_CORE_GENES, num_steps)
    field_dict = field_setup(protocol=field_protocol)
    app_field = None

    endpoint_dict = {}
    transfer_dict = {}
    proj_timeseries_array = np.zeros((len(CELLTYPE_LABELS), num_steps))
    basin_occupancy_timeseries = np.zeros((len(CELLTYPE_LABELS) + 1, num_steps), dtype=int)  # could have some spurious here too? not just last as mixed
    mixed_index = len(CELLTYPE_LABELS)  # i.e. last elem

    anneal_dict = anneal_setup(protocol=anneal_protocol)
    wandering = False

    for cell_idx in xrange(ensemble_idx, ensemble_idx + ensemble):
        if verbose:
            print "Simulating cell:", cell_idx
        cell = Cell(init_state, init_id)

        beta = anneal_dict['beta_start']  # reset beta to use in each trajectory

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
            proj_onto_init = projvec[CELLTYPE_ID[init_cond]]
            beta, wandering = anneal_iterate(proj_onto_init, beta, step, wandering, anneal_dict, verbose=verbose)

            # main call to update
            if step < num_steps:
                cell.update_state(beta=beta, app_field=None, fullstep_chunk=True)

        # update endpoints for each cell
        if projvec[topranked] > occ_threshold:
            endpoint_dict[cell_idx] = (CELLTYPE_LABELS[topranked], projvec[topranked])
        else:
            endpoint_dict[cell_idx] = ('mixed', projvec[topranked])

    return endpoint_dict, transfer_dict, proj_timeseries_array, basin_occupancy_timeseries


def fast_basin_stats(init_cond, init_state, init_id, ensemble, num_processes, num_steps=100, occ_threshold=0.7,
                     anneal_protocol=ANNEAL_PROTOCOL, field_protocol=FIELD_PROTOCOL, verbose=False):
    # prepare fn args and kwargs for wrapper
    kwargs_dict = {'num_steps': num_steps, 'anneal_protocol': anneal_protocol, 'field_protocol': field_protocol,
                   'occ_threshold': occ_threshold, 'verbose': verbose}
    fn_args_dict = [0]*num_processes
    if verbose:
        print "NUM_PROCESSES:", num_processes
    assert ensemble % num_processes == 0
    for i in xrange(num_processes):
        subensemble = ensemble / num_processes
        cell_startidx = i * subensemble
        if verbose:
            print "process:", i, "job size:", subensemble, "runs"
        fn_args_dict[i] = {'args': (init_cond, init_state, init_id, subensemble, cell_startidx),
                           'kwargs': kwargs_dict}
    # generate results list over workers
    t0 = time.time()
    pool = Pool(num_processes)
    results = pool.map(wrapper_get_basin_stats, fn_args_dict)
    pool.close()
    pool.join()
    if verbose:
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
    #check2 = np.sum(summed_basin_occupancy_timeseries, axis=0)
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


def ensemble_projection_timeseries(init_cond, ensemble, num_processes, num_steps=100, occ_threshold=0.7,
                                   anneal_protocol=ANNEAL_PROTOCOL, field_protocol=FIELD_PROTOCOL, plot=True):
    """
    Args:
    - init_cond: np array of init state OR string memory label
    - ensemble: ensemble of particles beginning at init_cond
    - num_steps: how many steps to iterate (each step updates every spin once)
    - occ_threshold: projection value cutoff to say state is in a basin (default: 0.7)
    - anneal_protocol: define how temperature changes during simulation
    What:
    - Track a timeseries of: ensemble mean projection onto each memory
    - Optionally plot
    Eeturn:
    - timeseries of projections onto store memories (dim p x T)
    """

    # prep io
    io_dict = run_subdir_setup(run_subfolder=ANALYSIS_SUBDIR)

    # generate initial state
    init_state, init_id = get_init_info(init_cond)

    # simulate ensemble - pooled wrapper call
    endpoint_dict, transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = \
        fast_basin_stats(init_cond, init_state, init_id, ensemble, num_processes, num_steps=num_steps,
                         anneal_protocol=anneal_protocol, field_protocol=field_protocol, occ_threshold=occ_threshold,
                         verbose=False)

    # normalize proj timeseries
    proj_timeseries_array = proj_timeseries_array / ensemble  # want ensemble average

    # save transition array and run info to file
    proj_timeseries_data = io_dict['datadir'] + os.sep + 'proj_proj_timeseries.txt'
    np.savetxt(proj_timeseries_data, proj_timeseries_array, delimiter=',')
    basin_occupancy_timeseries_data = io_dict['datadir'] + os.sep + 'proj_occupancy_timeseries.txt'
    np.savetxt(basin_occupancy_timeseries_data, basin_occupancy_timeseries, delimiter=',', fmt='%i')

    # plot output
    for idx in xrange(ensemble):
        if idx in transfer_dict:
            print idx, transfer_dict[idx], [CELLTYPE_LABELS[a] for a in transfer_dict[idx].keys()], endpoint_dict[idx]
        else:
            print idx, "no transfer dict entry", endpoint_dict[idx]
    if plot:
        highlights_CLPside = {6:'k', 8: 'blue', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
        savepath_proj = io_dict['plotdir'] + os.sep + 'proj_proj_timeseries.png'
        plot_proj_timeseries(proj_timeseries_array, num_steps, ensemble, savepath_proj, highlights=highlights_CLPside)
        savepath_occ = io_dict['plotdir'] + os.sep + 'proj_occupancy_timeseries.png'
        plot_basin_occupancy_timeseries(basin_occupancy_timeseries, num_steps, ensemble, occ_threshold, savepath_occ, highlights=highlights_CLPside)
        savepath_endpt = io_dict['plotdir'] + os.sep + 'endpt_stats.png'
        plot_basin_endpoints(endpoint_dict, num_steps, ensemble, savepath_endpt, highlights=highlights_CLPside)
    return proj_timeseries_array, basin_occupancy_timeseries, io_dict


def basin_transitions(init_cond, ensemble, num_steps, beta):
    # TODO note analysis basin grid fulfills this functionality, not great spurious handling though
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
        # common: 'HSC' / 'Common Lymphoid Progenitor (CLP)' / 'Common Myeloid Progenitor (CMP)' /
        #         'Megakaryocyte-Erythroid Progenitor (MEP)' / 'Granulocyte-Monocyte Progenitor (GMP)' / 'thymocyte DN'
        #         'thymocyte - DP' / 'neutrophils' / 'monocytes - classical'
        init_cond = 'HSC'  # note HSC index is 6 in mehta mems
        ensemble = 16
        num_steps = 50
        num_proc = cpu_count() / 2  # seems best to use only physical core count (1 core ~ 3x slower than 4)
        anneal_protocol = "constant"
        field_protocol = None

        t0 = time.time()
        _, _, io_dict = ensemble_projection_timeseries(init_cond, ensemble, num_proc, num_steps=num_steps,
                                                       occ_threshold=OCC_THRESHOLD, anneal_protocol=anneal_protocol,
                                                       plot=True)
        t1 = time.time() - t0

        # add info to run info file
        # TODO maybe move this INTO the function?
        info_list = [['fncall', 'ensemble_projection_timeseries()'], ['init_cond', init_cond], ['ensemble', ensemble],
                     ['num_steps', num_steps], ['num_proc', num_proc], ['anneal_protocol', anneal_protocol],
                     ['occ_threshold', OCC_THRESHOLD], ['field_protocol', field_protocol], ['time', t1]]
        runinfo_append(io_dict, info_list, multi=True)

        # less simple analysis
        # basin_transitions()

    # direct data plotting
    if plot_isolated_data:
        loaddata = np.loadtxt(RUNS_FOLDER +os.sep + 'proj_timeseries.txt', delimiter=',')
        ensemble = 100
        highlights_simple = {6:'k', 8: 'blue', 10: 'steelblue'}
        highlights_CLPside = {6:'k', 8: 'blue', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
        highlights_both = {6:'k', 8: 'blue', 10: 'steelblue', 9: 'forestgreen', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
        plot_proj_timeseries(loaddata, loaddata.shape[1], ensemble, RUNS_FOLDER + os.sep + 'proj_timeseries.png',
                             highlights=highlights_CLPside)
