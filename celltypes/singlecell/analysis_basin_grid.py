import numpy as np
import os
import time
from multiprocessing import cpu_count

from analysis_basin_plotting import plot_basin_grid
from analysis_basin_transitions import ensemble_projection_timeseries, get_basin_stats, fast_basin_stats, get_init_info, \
                                       ANNEAL_PROTOCOL, FIELD_PROTOCOL, ANALYSIS_SUBDIR, SPURIOUS_LIST, OCC_THRESHOLD, \
                                       save_and_plot_basinstats, load_basinstats, fetch_from_run_info
from singlecell_constants import RUNS_FOLDER, ASYNC_BATCH
from singlecell_data_io import run_subdir_setup, runinfo_append
from singlecell_simsetup import singlecell_simsetup


def gen_basin_grid(ensemble, num_processes, simsetup=None, num_steps=100, anneal_protocol=ANNEAL_PROTOCOL,
                   field_protocol=FIELD_PROTOCOL, occ_threshold=OCC_THRESHOLD, async_batch=ASYNC_BATCH, saveall=False,
                   save=True, plot=False, verbose=False, parallel=True):
    """
    generate matrix G_ij of size p x (p + k): grid of data between 0 and 1
    each row represents one of the p encoded basins as an initial condition
    each column represents an endpoint of the simulation starting at a given basin (row)
    G_ij would represent: starting in cell type i, G_ij of the ensemble transitioned to cell type j
    """
    # simsetup unpack for labelling plots
    if simsetup is None:
        simsetup = singlecell_simsetup()
    celltype_labels = simsetup['CELLTYPE_LABELS']

    io_dict = run_subdir_setup(run_subfolder=ANALYSIS_SUBDIR)
    basin_grid = np.zeros((len(celltype_labels), len(celltype_labels)+len(SPURIOUS_LIST)))
    for idx, celltype in enumerate(celltype_labels):
        print "Generating row: %d, %s" % (idx, celltype)
        if saveall:
            assert parallel
            plot_all = False
            proj_timeseries_array, basin_occupancy_timeseries, _, _ = \
                ensemble_projection_timeseries(celltype, ensemble, num_proc, simsetup=simsetup, num_steps=num_steps,
                                               anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                               occ_threshold=occ_threshold, async_batch=async_batch,
                                               plot=False, output=False)
            save_and_plot_basinstats(io_dict, proj_timeseries_array, basin_occupancy_timeseries, num_steps, ensemble,
                                     simsetup=simsetup, prefix=celltype, occ_threshold=occ_threshold, plot=plot_all)
        else:
            init_state, init_id = get_init_info(celltype, simsetup)
            if parallel:
                transfer_dict, proj_timeseries_array, basin_occupancy_timeseries, _ = \
                    fast_basin_stats(celltype, init_state, init_id, ensemble, num_processes, simsetup=simsetup,
                                     num_steps=num_steps, anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                     occ_threshold=occ_threshold, async_batch=async_batch, verbose=verbose)
            else:
                # Unparallelized for testing/profiling:
                transfer_dict, proj_timeseries_array, basin_occupancy_timeseries, _ = \
                    get_basin_stats(celltype, init_state, init_id, ensemble, 0, simsetup, num_steps=num_steps,
                                    anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                    async_batch=async_batch, occ_threshold=occ_threshold, verbose=verbose)
                proj_timeseries_array = proj_timeseries_array / ensemble  # ensure normalized (get basin stats won't do this)
        # fill in row of grid data from each celltype simulation
        basin_grid[idx, :] = basin_occupancy_timeseries[:,-1]
    if save:
        np.savetxt(io_dict['latticedir'] + os.sep + 'gen_basin_grid.txt', basin_grid, delimiter=',', fmt='%.4f')
    if plot:
        plot_basin_grid(basin_grid, ensemble, num_steps, celltype_labels, io_dict['latticedir'], SPURIOUS_LIST)
    return basin_grid, io_dict


def load_basin_grid(filestr_data):
    # TODO: prepare IO functions for standardized sim settings dict struct
    basin_grid = np.loadtxt(filestr_data, delimiter=',', dtype=float)
    #sim_settings = load_sim_settings(filestr_settings)
    return basin_grid


def grid_stats(grid_data, printtorank=10):
    """
    Prints info based on row statistics of the grid
    Args:
        grid_data: basin grid data from basin occupancy hopping sim
    Returns:
        None
    """
    basin_row_sum = np.sum(grid_data, axis=1)
    ensemble = basin_row_sum[0]
    ref_list = celltype_labels + SPURIOUS_LIST
    for row in xrange(len(celltype_labels)):
        assert basin_row_sum[row] == ensemble  # make sure all rows sum to expected value
        sortedmems_smalltobig = np.argsort(grid_data[row, :])
        sortedmems_bigtosmall = sortedmems_smalltobig[::-1]
        print "\nRankings for row", row, celltype_labels[row], "(sum %d)" % int(basin_row_sum[row])
        for rank in xrange(printtorank):
            ranked_col_idx = sortedmems_bigtosmall[rank]
            ranked_label = ref_list[ranked_col_idx]
            print rank, ranked_label, grid_data[row, ranked_col_idx], grid_data[row, ranked_col_idx] / ensemble


def grid_video(rundir, vidname, imagedir=None, ext='.mp4', fps=20):
    """
    Make a video of the grid over time using ffmpeg.
    Note: ffmpeg must be installed and on system path.
    Args:
        rundir: Assumes sequentially named images of grid over time are in "plot_lattice" subdir of rundir.
        vidname: filename for the video (no extension); it will be placed in a "video" subdir of rundir
        imagedir: override use of "plot_lattice" subdir of rundir as the source [Default: None]
        ext: only '.mp4' has been tested, seems to work on Windows Media Player but not VLC
        fps: video frames per second; 1, 5, 20 work well
    Returns:
        path to video
    """
    from utils.make_video import make_video_ffmpeg
    # args specify
    if imagedir is None:
        imagedir = rundir + os.sep + "plot_lattice"
    if not os.path.exists(rundir + os.sep + "video"):
        os.makedirs(rundir + os.sep + "video")
    videopath = rundir + os.sep + "video" + os.sep + vidname + ext
    # call make video fn
    print "Creating video at %s..." % videopath
    make_video_ffmpeg(imagedir, videopath, fps=fps, ffmpeg_dir=None)
    print "Done"
    return videopath


if __name__ == '__main__':
    run_basin_grid = False
    load_and_plot_basin_grid = True
    reanalyze_grid_over_time = False
    make_grid_video = False
    print_grid_stats_from_file = False

    # prep simulation globals
    simsetup = singlecell_simsetup()
    celltype_labels = simsetup['CELLTYPE_LABELS']

    if run_basin_grid:
        ensemble = 1000
        timesteps = 500
        field_protocol = FIELD_PROTOCOL
        anneal_protocol = ANNEAL_PROTOCOL
        num_proc = cpu_count() / 2
        async_batch = True
        plot = False
        saveall = True
        parallel = True

        # run gen_basin_grid
        t0 = time.time()
        basin_grid, io_dict = gen_basin_grid(ensemble, num_proc, simsetup=simsetup, num_steps=timesteps,
                                             anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                             async_batch=async_batch, saveall=saveall, plot=plot, parallel=parallel)
        t1 = time.time() - t0
        print "GRID TIMER:", t1

        # add info to run info file TODO maybe move this INTO the function?
        info_list = [['fncall', 'gen_basin_grid()'], ['ensemble', ensemble], ['num_steps', timesteps],
                     ['num_proc', num_proc], ['anneal_protocol', anneal_protocol], ['field_protocol', field_protocol],
                     ['occ_threshold', OCC_THRESHOLD], ['async_batch', async_batch], ['time', t1]]
        runinfo_append(io_dict, info_list, multi=True)

    # direct data plotting
    if load_and_plot_basin_grid:
        # specify paths and load data / parameters
        rundir = RUNS_FOLDER + os.sep + ANALYSIS_SUBDIR + os.sep + "grid_335324_5kx200_fieldYama"
        latticedir = rundir + os.sep + "lattice"
        filestr_data = latticedir + os.sep + "gen_basin_grid.txt"
        basin_grid_data = load_basin_grid(filestr_data)
        ensemble, num_steps = fetch_from_run_info(rundir + os.sep + 'run_info.txt', ['ensemble', 'num_steps'])
        # build grid plots
        plot_basin_grid(basin_grid_data, ensemble, num_steps, celltype_labels, latticedir, SPURIOUS_LIST,
                        relmax=False, ext='.pdf', vforce=0.5)
        plot_basin_grid(basin_grid_data, ensemble, num_steps, celltype_labels, latticedir, SPURIOUS_LIST,
                        relmax=False, ext='.pdf', vforce=1.0)

    # use labelled collection of timeseries from each row to generate multiple grids over time
    if reanalyze_grid_over_time:
        # step 0 specify ensemble, num steps, and location of row data
        rundir = RUNS_FOLDER + os.sep + ANALYSIS_SUBDIR + os.sep + "aug11 - 1000ens x 500step - fullRandomSteps"
        ensemble, num_steps = fetch_from_run_info(rundir + os.sep + 'run_info.txt', ['ensemble', 'num_steps'])
        # step 1 restructure data
        rowdatadir = rundir + os.sep + "data"
        latticedir = rundir + os.sep + "lattice"
        plotlatticedir = rundir + os.sep + "plot_lattice"
        p = len(celltype_labels)
        k = len(SPURIOUS_LIST)
        grid_over_time = np.zeros((p, p+k, num_steps))
        for idx, celltype in enumerate(celltype_labels):
            print "loading:", idx, celltype
            proj_timeseries_array, basin_occupancy_timeseries = load_basinstats(rowdatadir, celltype)
            grid_over_time[idx, :, :] += basin_occupancy_timeseries
        # step 2 save and plot
        vforce = 0.5
        filename = 'grid_at_step'
        for step in xrange(num_steps):
            print "step", step
            grid_at_step = grid_over_time[:, :, step]
            namemod = '_%d' % step
            np.savetxt(latticedir + os.sep + filename + namemod + '.txt', grid_at_step, delimiter=',', fmt='%.4f')
            plot_basin_grid(grid_at_step, ensemble, step, celltype_labels, plotlatticedir, SPURIOUS_LIST,
                            plotname=filename, relmax=False, vforce=vforce, namemod=namemod, ext='.jpg')

    if make_grid_video:
        custom_fps = 5  # 1, 5, or 20 are good
        rundir = RUNS_FOLDER + os.sep + ANALYSIS_SUBDIR + os.sep + "aug11 - 1000ens x 500step - fullRandomSteps"
        vidname = "grid_1000x500_vmax1.0_stepFullyRandom_fps%d" % custom_fps
        latticedir = rundir + os.sep + "ARCHIVE_partial_vforce1.00_plot_lattice"
        videopath = grid_video(rundir, vidname, imagedir=latticedir, fps=custom_fps)

    if print_grid_stats_from_file:
        filestr_data = RUNS_FOLDER + os.sep + "gen_basin_grid_C.txt"
        basin_grid_data = load_basin_grid(filestr_data)
        grid_stats(basin_grid_data)
        """
        ensemble = 960
        basin_grid_A = load_basin_grid(RUNS_FOLDER + os.sep + "gen_basin_grid_A.txt") / ensemble
        basin_grid_B = load_basin_grid(RUNS_FOLDER + os.sep + "gen_basin_grid_B.txt") / ensemble
        basin_grid_C = load_basin_grid(RUNS_FOLDER + os.sep + "gen_basin_grid_C.txt") / ensemble
        basin_grid_D = load_basin_grid(RUNS_FOLDER + os.sep + "gen_basin_grid_D.txt") / ensemble
        basin_grid_E = load_basin_grid(RUNS_FOLDER + os.sep + "gen_basin_grid_E.txt") / ensemble
        for idx, label in enumerate(celltype_labels):
            print idx, "%.2f vs %.2f" % (basin_grid_A[idx,-1], basin_grid_C[idx,-1]), label
            print idx, "%.2f vs %.2f" % (basin_grid_B[idx,-1], basin_grid_D[idx,-1]), label
        """
