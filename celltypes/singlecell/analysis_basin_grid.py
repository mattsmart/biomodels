import numpy as np
import os
import time
from multiprocessing import cpu_count

from analysis_basin_plotting import plot_basin_grid
from analysis_basin_transitions import ensemble_projection_timeseries, fast_basin_stats, get_init_info, OCC_THRESHOLD, \
                                       ANNEAL_PROTOCOL, FIELD_PROTOCOL, ANALYSIS_SUBDIR, SPURIOUS_LIST, \
                                       save_and_plot_basinstats, load_basinstats
from singlecell_constants import RUNS_FOLDER
from singlecell_data_io import run_subdir_setup, runinfo_append
from singlecell_simsetup import CELLTYPE_LABELS


def gen_basin_grid(ensemble, num_processes, num_steps=100, anneal_protocol=ANNEAL_PROTOCOL,
                   field_protocol=FIELD_PROTOCOL, occ_threshold=OCC_THRESHOLD, saveall=False, save=True,
                   plot=False, verbose=False):
    """
    generate matrix G_ij of size p x (p + k): grid of data between 0 and 1
    each row represents one of the p encoded basins as an initial condition
    each column represents an endpoint of the simulation starting at a given basin (row)
    G_ij would represent: starting in cell type i, G_ij of the ensemble transitioned to cell type j
    """
    io_dict = run_subdir_setup(run_subfolder=ANALYSIS_SUBDIR)
    basin_grid = np.zeros((len(CELLTYPE_LABELS), len(CELLTYPE_LABELS)+len(SPURIOUS_LIST)))
    for idx, celltype in enumerate(CELLTYPE_LABELS):
        if verbose:
            print "Generating row: %d, %s" % (idx, celltype)
        if saveall:
            plot_all = False
            proj_timeseries_array, basin_occupancy_timeseries, _ = \
                ensemble_projection_timeseries(celltype, ensemble, num_proc, num_steps=num_steps,
                                               anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                               occ_threshold=occ_threshold, plot=False, output=False)
            save_and_plot_basinstats(io_dict, proj_timeseries_array, basin_occupancy_timeseries, num_steps, ensemble,
                                     prefix=celltype, occ_threshold=occ_threshold, plot=plot_all)
        else:
            init_state, init_id = get_init_info(celltype)
            transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = \
                fast_basin_stats(celltype, init_state, init_id, ensemble, num_processes, num_steps=num_steps,
                                 anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                 occ_threshold=occ_threshold, verbose=verbose)
            """ Unparallelized for testing/profiling:
            transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = \
                get_basin_stats(celltype, init_state, init_id, ensemble, 0, num_steps=num_steps, 
                                anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                occ_threshold=occ_threshold, verbose=verbose)
            """
        # fill in row of grid data from each celltype simulation
        basin_grid[idx, :] = basin_occupancy_timeseries[:,-1]
    if save:
        np.savetxt(io_dict['latticedir'] + os.sep + 'gen_basin_grid.txt', basin_grid, delimiter=',', fmt='%.4f')
    if plot:
        plot_basin_grid(basin_grid, ensemble, num_steps, io_dict['latticedir'], SPURIOUS_LIST)
    return basin_grid, io_dict


def load_basin_grid(filestr_data):
    # TODO: prepare IO functions for standardized sim settings dict struct
    basin_grid = np.loadtxt(filestr_data, delimiter=',', dtype=float)
    #sim_settings = load_sim_settings(filestr_settings)
    return basin_grid


if __name__ == '__main__':
    run_basin_grid = False
    load_and_plot_basin_grid = True
    reanalyze_grid_over_time = False

    if run_basin_grid:
        # TODO: find way to prevent reloading the interaction info from singlcell_simsetup
        ensemble = 40
        timesteps = 4
        field_protocol = FIELD_PROTOCOL
        anneal_protocol = ANNEAL_PROTOCOL
        num_proc = cpu_count() / 2
        plot = False
        saveall = True

        # run gen_basin_grid
        t0 = time.time()
        basin_grid, io_dict = gen_basin_grid(ensemble, num_proc, num_steps=timesteps, anneal_protocol=anneal_protocol,
                                             field_protocol=field_protocol, saveall=saveall, plot=plot)
        t1 = time.time() - t0
        print "GRID TIMER:", t1

        # add info to run info file TODO maybe move this INTO the function?
        info_list = [['fncall', 'gen_basin_grid()'], ['ensemble', ensemble], ['num_steps', timesteps],
                     ['num_proc', num_proc], ['anneal_protocol', anneal_protocol], ['field_protocol', field_protocol],
                     ['occ_threshold', OCC_THRESHOLD], ['time', t1]]
        runinfo_append(io_dict, info_list, multi=True)

    # direct data plotting
    if load_and_plot_basin_grid:
        filestr_data = RUNS_FOLDER + os.sep + 'gen_basin_grid.txt'
        basin_grid_data = load_basin_grid(filestr_data)
        ensemble = 40
        num_steps = 200
        plot_basin_grid(basin_grid_data, ensemble, num_steps, RUNS_FOLDER, SPURIOUS_LIST)

    # use labelled collection of timeseries from each row to generate multiple grids over time
    if reanalyze_grid_over_time:
        # step 0 specify ensemble, num steps, and location of row data
        ensemble = 40
        num_steps = 200
        rundir = RUNS_FOLDER + os.sep + ANALYSIS_SUBDIR + os.sep + "grid_40x200_save_rowdata"
        # step 1 restructure data
        rowdatadir = rundir + os.sep + "data"
        latticedir = rundir + os.sep + "plot_lattice"
        p = len(CELLTYPE_LABELS)
        k = len(SPURIOUS_LIST)
        grid_over_time = np.zeros((p, p+k, num_steps))
        for idx, celltype in enumerate(CELLTYPE_LABELS):
            print "loading:", idx, celltype
            proj_timeseries_array, basin_occupancy_timeseries = load_basinstats(rowdatadir, celltype)
            grid_over_time[idx, :, :] += basin_occupancy_timeseries
        # step 2 save and plot
        for step in xrange(num_steps):
            print "step", step
            grid_at_step = grid_over_time[:, :, step]
            filename = 'grid_at_step_%d' % step
            np.savetxt(latticedir + os.sep + filename + '.txt', grid_at_step, delimiter=',', fmt='%.4f')
            plot_basin_grid(grid_at_step, ensemble, step, latticedir, SPURIOUS_LIST, plotname=filename, relmax=False)