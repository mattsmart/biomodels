import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count

from analysis_basin_plotting import plot_proj_timeseries, plot_basin_occupancy_timeseries, plot_basin_step
from singlecell_class import Cell
from singlecell_constants import RUNS_FOLDER, IPSC_CORE_GENES_EFFECTS
from singlecell_data_io import run_subdir_setup, runinfo_append
from singlecell_functions import construct_app_field_from_genes
from singlecell_simsetup import singlecell_simsetup, unpack_simsetup


# analysis settings
ANALYSIS_SUBDIR = "basin_transitions"
ANNEAL_BETA = 1.3
ANNEAL_PROTOCOL = "protocol_A"
FIELD_PROTOCOL = None
OCC_THRESHOLD = 0.7
SPURIOUS_LIST = ["mixed"]

# analysis plotting
highlights_CLPside = {6: 'k', 8: 'blue', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
highlights_simple = {6: 'k', 8: 'blue', 10: 'steelblue'}
highlights_both = {6: 'k', 8: 'blue', 10: 'steelblue', 9: 'forestgreen', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
DEFAULT_HIGHLIGHTS = highlights_CLPside


def field_setup(gene_id, protocol=FIELD_PROTOCOL):
    """
    Construct applied field vector (either fixed or on varying under a field protocol) to bias the dynamics

    Notes on named fields
    Yamanaka factor (OSKM + nanog) names in mehta datafile: Sox2, Pou5f1 (oct3/4), Klf4, Mycbp, nanog
    """
    # TODO build
    assert protocol in [None]
    field_dict = {'protocol': protocol,
                   'field_start': None}
    app_field_start = construct_app_field_from_genes(IPSC_CORE_GENES_EFFECTS, gene_id, num_steps=0)
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


def get_init_info(init_cond, simsetup):
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
        init_state = simsetup['XI'][:, simsetup['CELLTYPE_ID'][init_cond]]
        init_id = init_cond
    return init_state, init_id


def save_and_plot_basinstats(io_dict, proj_data, occ_data, num_steps, ensemble, prefix='', simsetup=None,
                             occ_threshold=OCC_THRESHOLD, plot=True, highlights=DEFAULT_HIGHLIGHTS):
    # filename prep
    if prefix[-1] != '_':
        prefix += '_'
    # simsetup unpack for labelling plots
    if simsetup is None:
        simsetup = singlecell_simsetup()
    memory_labels = simsetup['CELLTYPE_LABELS']
    memory_id = simsetup['CELLTYPE_ID']
    N, P, gene_labels, memory_labels, gene_id, celltype_id, xi, _, a_inv, intxn_matrix, _ = unpack_simsetup(simsetup)
    # path setup
    datapath_proj = io_dict['datadir'] + os.sep + '%sproj_timeseries.txt' % prefix
    datapath_occ = io_dict['datadir'] + os.sep + '%soccupancy_timeseries.txt' % prefix
    plotpath_proj = io_dict['plotdir'] + os.sep + '%sproj_timeseries.png' % prefix
    plotpath_occ = io_dict['plotdir'] + os.sep + '%soccupancy_timeseries.png' % prefix
    plotpath_basin_endpt = io_dict['plotdir'] + os.sep + '%sendpt_distro.png' % prefix
    # save data to file
    np.savetxt(datapath_proj, proj_data, delimiter=',', fmt='%i')
    np.savetxt(datapath_occ, occ_data, delimiter=',', fmt='%i')
    # plot and save figs
    if plot:
        plot_proj_timeseries(proj_data, num_steps, ensemble, memory_labels, plotpath_proj, highlights=highlights)
        plot_basin_occupancy_timeseries(occ_data, num_steps, ensemble, memory_labels, occ_threshold, SPURIOUS_LIST, plotpath_occ, highlights=highlights)
        plot_basin_step(occ_data[:, -1], num_steps, ensemble, memory_labels, memory_id, SPURIOUS_LIST, plotpath_basin_endpt, highlights=highlights)
    return


def load_basinstats(rowdata_dir, celltype):
    proj_name = "%s_proj_timeseries.txt" % celltype
    occ_name = "%s_occupancy_timeseries.txt" % celltype
    proj_timeseries_array = np.loadtxt(rowdata_dir + os.sep + proj_name, delimiter=',', dtype=float)
    basin_occupancy_timeseries = np.loadtxt(rowdata_dir + os.sep + occ_name, delimiter=',', dtype=int)
    return proj_timeseries_array, basin_occupancy_timeseries


def wrapper_get_basin_stats(fn_args_dict):
    np.random.seed()
    if fn_args_dict['kwargs'] is not None:
        return get_basin_stats(*fn_args_dict['args'], **fn_args_dict['kwargs'])
    else:
        return get_basin_stats(*fn_args_dict['args'])


def get_basin_stats(init_cond, init_state, init_id, ensemble, ensemble_idx, simsetup, num_steps=100,
                    anneal_protocol=ANNEAL_PROTOCOL, field_protocol=FIELD_PROTOCOL, occ_threshold=OCC_THRESHOLD, verbose=False):

    # simsetup unpack
    N, _, gene_labels, memory_labels, gene_id, celltype_id, xi, _, a_inv, intxn_matrix, _ = unpack_simsetup(simsetup)

    # prep applied field TODO: how to include applied field neatly
    field_dict = field_setup(simsetup['GENE_ID'], protocol=field_protocol)
    app_field = None

    transfer_dict = {}
    proj_timeseries_array = np.zeros((len(memory_labels), num_steps))
    basin_occupancy_timeseries = np.zeros((len(memory_labels) + len(SPURIOUS_LIST), num_steps), dtype=int)
    assert len(SPURIOUS_LIST) == 1
    mixed_index = len(memory_labels)  # i.e. last elem

    anneal_dict = anneal_setup(protocol=anneal_protocol)
    wandering = False

    for cell_idx in xrange(ensemble_idx, ensemble_idx + ensemble):
        if verbose:
            print "Simulating cell:", cell_idx
        cell = Cell(init_state, init_id, memory_labels, gene_labels)

        beta = anneal_dict['beta_start']  # reset beta to use in each trajectory

        for step in xrange(num_steps):

            # report on each mem proj ranked
            projvec = cell.get_memories_projection(a_inv, N, xi)
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
                    ranked_mem = memory_labels[ranked_mem_idx]
                    print rank, ranked_mem_idx, ranked_mem, projvec[ranked_mem_idx], absprojvec[ranked_mem_idx]

            if projvec[topranked] > occ_threshold:
                basin_occupancy_timeseries[topranked, step] += 1
            else:
                basin_occupancy_timeseries[mixed_index, step] += 1

            if topranked != celltype_id[init_cond] and projvec[topranked] > occ_threshold:
                if cell_idx not in transfer_dict:
                    transfer_dict[cell_idx] = {topranked: (step, projvec[topranked])}
                else:
                    if topranked not in transfer_dict[cell_idx]:
                        transfer_dict[cell_idx] = {topranked: (step, projvec[topranked])}

            # annealing block
            proj_onto_init = projvec[celltype_id[init_cond]]
            beta, wandering = anneal_iterate(proj_onto_init, beta, step, wandering, anneal_dict, verbose=verbose)

            # main call to update
            if step < num_steps:
                cell.update_state(intxn_matrix, beta=beta, app_field=None, fullstep_chunk=True)

    return transfer_dict, proj_timeseries_array, basin_occupancy_timeseries


def fast_basin_stats(init_cond, init_state, init_id, ensemble, num_processes, simsetup=None, num_steps=100, occ_threshold=0.7,
                     anneal_protocol=ANNEAL_PROTOCOL, field_protocol=FIELD_PROTOCOL, verbose=False):
    # simsetup unpack
    if simsetup is None:
        simsetup = singlecell_simsetup()
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
        fn_args_dict[i] = {'args': (init_cond, init_state, init_id, subensemble, cell_startidx, simsetup),
                           'kwargs': kwargs_dict}
    # generate results list over workers
    t0 = time.time()
    pool = Pool(num_processes)
    print "pooling"
    results = pool.map(wrapper_get_basin_stats, fn_args_dict)
    print "done"
    pool.close()
    pool.join()
    if verbose:
        print "TIMER:", time.time() - t0
    # collect pooled results
    summed_transfer_dict = {}  # TODO remove?
    summed_proj_timeseries_array = np.zeros((len(simsetup['CELLTYPE_LABELS']), num_steps))
    summed_basin_occupancy_timeseries = np.zeros((len(simsetup['CELLTYPE_LABELS']) + 1, num_steps), dtype=int)  # could have some spurious here too? not just last as mixed
    for i, result in enumerate(results):
        transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = result
        summed_transfer_dict.update(transfer_dict)  # TODO check
        summed_proj_timeseries_array += proj_timeseries_array
        summed_basin_occupancy_timeseries += basin_occupancy_timeseries
    #check2 = np.sum(summed_basin_occupancy_timeseries, axis=0)

    # notmalize proj timeseries
    summed_proj_timeseries_array = summed_proj_timeseries_array / ensemble  # want ensemble average

    return summed_transfer_dict, summed_proj_timeseries_array, summed_basin_occupancy_timeseries


def ensemble_projection_timeseries(init_cond, ensemble, num_processes, simsetup=None, num_steps=100, occ_threshold=0.7,
                                   anneal_protocol=ANNEAL_PROTOCOL, field_protocol=FIELD_PROTOCOL,
                                   output=True, plot=True):
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

    # simsetup unpack
    if simsetup is None:
        simsetup = singlecell_simsetup()

    # prep io
    if output:
        io_dict = run_subdir_setup(run_subfolder=ANALYSIS_SUBDIR)
    else:
        assert not plot
        io_dict = None

    # generate initial state
    init_state, init_id = get_init_info(init_cond, simsetup)

    # simulate ensemble - pooled wrapper call
    transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = \
        fast_basin_stats(init_cond, init_state, init_id, ensemble, num_processes, simsetup=simsetup, num_steps=num_steps,
                         anneal_protocol=anneal_protocol, field_protocol=field_protocol, occ_threshold=occ_threshold,
                         verbose=False)

    # save data and plot figures
    if output:
        save_and_plot_basinstats(io_dict, proj_timeseries_array, basin_occupancy_timeseries, num_steps, ensemble,
                                 simsetup=simsetup, prefix=init_id, occ_threshold=occ_threshold, plot=plot)
    # plot output
    for idx in xrange(ensemble):
        if idx in transfer_dict:
            print idx, transfer_dict[idx], [simsetup['CELLTYPE_LABELS'][a] for a in transfer_dict[idx].keys()]
    return proj_timeseries_array, basin_occupancy_timeseries, io_dict


def basin_transitions(init_cond, ensemble, num_steps, beta, simsetup):
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
    basins_dim = len(simsetup['CELLTYPE_LABELS']) + 1
    spurious_index = len(simsetup['CELLTYPE_LABELS'])
    transition_data = np.zeros((basins_dim, basins_dim))

    for idx, memory_label in enumerate(simsetup['CELLTYPE_LABELS']):
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

    # prep simulation globals
    simsetup = singlecell_simsetup()

    if gen_basin_data:
        # common: 'HSC' / 'Common Lymphoid Progenitor (CLP)' / 'Common Myeloid Progenitor (CMP)' /
        #         'Megakaryocyte-Erythroid Progenitor (MEP)' / 'Granulocyte-Monocyte Progenitor (GMP)' / 'thymocyte DN'
        #         'thymocyte - DP' / 'neutrophils' / 'monocytes - classical'
        init_cond = 'HSC'  # note HSC index is 6 in mehta mems
        ensemble = 16
        num_steps = 100
        num_proc = cpu_count() / 2  # seems best to use only physical core count (1 core ~ 3x slower than 4)
        anneal_protocol = "protocol_A"
        field_protocol = None
        plot = False
        parallel = False

        # run and time basin ensemble sim
        t0 = time.time()
        if parallel:
            _, _, io_dict = ensemble_projection_timeseries(init_cond, ensemble, num_proc, num_steps=num_steps,
                                                           simsetup=simsetup, occ_threshold=OCC_THRESHOLD,
                                                           anneal_protocol=anneal_protocol, plot=plot)
        else:
            # Unparallelized for testing/profiling:
            init_state, init_id = get_init_info(init_cond, simsetup)
            io_dict = run_subdir_setup(run_subfolder=ANALYSIS_SUBDIR)
            transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = \
                get_basin_stats(init_cond, init_state, init_id, ensemble, 0, simsetup, num_steps=num_steps,
                                anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                occ_threshold=OCC_THRESHOLD, verbose=True)
            proj_timeseries_array = proj_timeseries_array / ensemble  # ensure normalized (get basin stats won't do this)
        t1 = time.time() - t0
        print "Runtime:", t1

        # append info to run info file  TODO maybe move this INTO the function?
        info_list = [['fncall', 'ensemble_projection_timeseries()'], ['init_cond', init_cond], ['ensemble', ensemble],
                     ['num_steps', num_steps], ['num_proc', num_proc], ['anneal_protocol', anneal_protocol],
                     ['occ_threshold', OCC_THRESHOLD], ['field_protocol', field_protocol], ['time', t1]]
        runinfo_append(io_dict, info_list, multi=True)

    # direct data plotting
    if plot_isolated_data:
        loaddata = np.loadtxt(RUNS_FOLDER + os.sep + 'proj_timeseries.txt', delimiter=',')
        ensemble = 100
        plot_proj_timeseries(loaddata, loaddata.shape[1], ensemble, simsetup['CELLTYPE_LABELS'],
                             RUNS_FOLDER + os.sep + 'proj_timeseries.png', highlights=highlights_CLPside)
