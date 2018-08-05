import numpy as np
import os
import time
from multiprocessing import cpu_count

from analysis_basin_plotting import plot_basin_grid
from analysis_basin_transitions import ensemble_projection_timeseries, fast_basin_stats, get_init_info, OCC_THRESHOLD, \
                                       ANNEAL_PROTOCOL, FIELD_PROTOCOL, ANALYSIS_SUBDIR, save_and_plot_basinstats
from singlecell_constants import RUNS_FOLDER
from singlecell_data_io import run_subdir_setup, runinfo_append
from singlecell_simsetup import CELLTYPE_LABELS


def gen_basin_grid(ensemble, num_processes, num_steps=100, anneal_protocol=ANNEAL_PROTOCOL,
                   field_protocol=FIELD_PROTOCOL, occ_threshold=OCC_THRESHOLD, k=1, saveall=False, save=True,
                   plot=False, verbose=False):
    """
    generate matrix G_ij of size p x (p + k): grid of data between 0 and 1
    each row represents one of the p encoded basins as an initial condition
    each column represents an endpoint of the simulation starting at a given basin (row)
    G_ij would represent: starting in cell type i, G_ij of the ensemble transitioned to cell type j
    k represents the number of extra tracked states, by default this is 1 (i.e. mixed state, not in any basin)
    """

    #TODO store output of each basin sim in own dir using override dir arg passed to ensemble fn
    io_dict = run_subdir_setup(run_subfolder=ANALYSIS_SUBDIR)

    basin_grid = np.zeros((len(CELLTYPE_LABELS), len(CELLTYPE_LABELS)+k))
    for idx, celltype in enumerate(CELLTYPE_LABELS):

        if verbose:
            print "Generating row: %d, %s" % (idx, celltype)

        if saveall:
            # TODO adjust this as above
            proj_timeseries_array, basin_occupancy_timeseries, _ = \
                ensemble_projection_timeseries(celltype, ensemble, num_proc, num_steps=num_steps,
                                               anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                               occ_threshold=occ_threshold, plot=False, output=False)
            save_and_plot_basinstats(io_dict, proj_timeseries_array, basin_occupancy_timeseries, num_steps, ensemble,
                                     prefix=celltype, occ_threshold=occ_threshold, plot=True)
        else:
            init_state, init_id = get_init_info(celltype)
            transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = \
                fast_basin_stats(celltype, init_state, init_id, ensemble, num_processes, num_steps=num_steps,
                                 anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                 occ_threshold=occ_threshold, verbose=False)
            """
            transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = \
                get_basin_stats(celltype, init_state, init_id, ensemble, 0, num_steps=20, 
                                anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                occ_threshold=OCC_THRESHOLD, verbose=False)
            """
        # fill in row of grid data from each celltype simulation
        basin_grid[idx, :] = basin_occupancy_timeseries[:,-1]
    if save:
        np.savetxt(io_dict['basedir'] + os.sep + 'gen_basin_grid.txt', basin_grid, delimiter=',', fmt='%.4f')
    if plot:
        plot_basin_grid(basin_grid, ensemble, num_steps, io_dict['basedir'], k=k)
    return basin_grid, io_dict


def load_basin_grid(filestr_data):
    # TODO: prepare IO functions for standardized sim settings dict struct
    basin_grid = np.loadtxt(filestr_data, delimiter=',', dtype=float)
    #sim_settings = load_sim_settings(filestr_settings)
    return basin_grid


if __name__ == '__main__':

    # TODO io settings propogate
    switch_gen_basin_grid = True

    if switch_gen_basin_grid:
        # TODO: find way to prevent reloading the interaction info from singlcell_simsetup
        ensemble = 4
        timesteps = 20
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
    else:
        filestr_data = RUNS_FOLDER + os.sep + 'gen_basin_grid.txt'
        basin_grid_data = load_basin_grid(filestr_data)
        ensemble = 960
        num_steps = 100
        plot_basin_grid(basin_grid_data, ensemble, num_steps, RUNS_FOLDER, k=1)
