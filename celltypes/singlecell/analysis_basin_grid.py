import matplotlib.pyplot as plt
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count

from analysis_basin_transitions import ensemble_projection_timeseries, fast_basin_stats, get_init_info, OCC_THRESHOLD, ANNEAL_BETA, get_basin_stats
from singlecell_constants import RUNS_FOLDER, IPSC_CORE_GENES, BETA
from singlecell_data_io import run_subdir_setup
from singlecell_simsetup import N, P, XI, CELLTYPE_ID, A_INV, J, GENE_ID, GENE_LABELS, CELLTYPE_LABELS


def gen_basin_grid(ensemble, num_processes, num_steps=100, beta=ANNEAL_BETA, occ_threshold=OCC_THRESHOLD,
                   k=1, saveall=False, save=True, plot=False):
    """
    generate matrix G_ij of size p x (p + k): grid of data between 0 and 1
    each row represents one of the p encoded basins as an initial condition
    each column represents an endpoint of the simulation starting at a given basin (row)
    G_ij would represent: starting in cell type i, G_ij of the ensemble transitioned to cell type j
    k represents the number of extra tracked states, by default this is 1 (i.e. mixed state, not in any basin)
    """

    #TODO store output of each basin sim in own dir using override dir arg passed to ensemble fn

    basin_grid = np.zeros((len(CELLTYPE_LABELS), len(CELLTYPE_LABELS)+k))
    for idx, celltype in enumerate(CELLTYPE_LABELS):
        if saveall:
            proj_timeseries_array, basin_occupancy_timeseries = \
                ensemble_projection_timeseries(celltype, ensemble, num_proc, num_steps=num_steps, beta=beta,
                                               occ_threshold=occ_threshold, plot=True, anneal=True)
        else:
            init_state, init_id = get_init_info(celltype)
            endpoint_dict, transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = \
                fast_basin_stats(celltype, init_state, init_id, ensemble, num_processes,
                                 num_steps=num_steps, beta=beta, anneal=True, verbose=False, occ_threshold=occ_threshold)
            """
            endpoint_dict, transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = \
                get_basin_stats(celltype, init_state, init_id, ensemble, 0, num_steps=20, beta=ANNEAL_BETA,
                                anneal=True,
                                verbose=False, occ_threshold=OCC_THRESHOLD)
            """
        # fill in row of grid data from eah celltype simulation
        basin_grid[idx, :] = basin_occupancy_timeseries[:,-1]
    if save:
        np.savetxt(RUNS_FOLDER + os.sep + 'gen_basin_grid.txt', basin_grid, delimiter=',', fmt='%.4f')
    if plot:
        plot_basin_grid(basin_grid, ensemble, num_steps, k=k)
    return basin_grid


def load_basin_grid(filestr_data):
    # TODO: prepare IO functions for standardized sim settings dict struct
    basin_grid = np.loadtxt(filestr_data, delimiter=',', dtype=float)
    #sim_settings = load_sim_settings(filestr_settings)
    return basin_grid


def plot_basin_grid(grid_data, ensemble, steps, k=1, ax=None, normalize=True, fs=10, relmax=True):
    """
    plot matrix G_ij of size p x (p + k): grid of data between 0 and 1
    each row represents one of the p encoded basins as an initial condition
    each column represents an endpoint of the simulation starting at a given basin (row)
    G_ij would represent: starting in cell type i, G_ij of the ensemble transitioned to cell type j
    Args:
    - relmax means max of color scale will be data max
    - k represents the number of extra tracked states, by default this is 1 (i.e. mixed state, not in any basin)
    """
    assert grid_data.shape == (len(CELLTYPE_LABELS), len(CELLTYPE_LABELS) + k)

    assert normalize
    datamax = np.max(grid_data)

    if np.max(grid_data) > 1.0 and normalize:
        grid_data = grid_data / ensemble
        datamax = datamax / ensemble

    if relmax:
        vmax = datamax
    else:
        if normalize:
            vmax = 1.0
        else:
            vmax = ensemble

    if not ax:
        ax = plt.gca()
        plt.gcf().set_size_inches(18.5, 12.5)
    # plot the heatmap
    imshow_kw = {'cmap': 'YlGnBu', 'vmin': 0.0, 'vmax': vmax}  # note: fix at 0.5 of max works nice
    im = ax.imshow(grid_data, **imshow_kw)
    # create colorbar
    cbar_kw = {'aspect': 30, 'pad': 0.02}   # larger aspect, thinner bar
    cbarlabel = 'Basin occupancy fraction'
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fs+2, labelpad=20)

    # hack title to bottom
    plt.text(0.5, 1.3, 'Basin grid transition data (%d cells per basin, %d steps)' % (ensemble, steps),
             horizontalalignment='center', transform=ax.transAxes, fontsize=fs+4)
    # axis labels
    plt.xlabel('Ensemble fraction after %d steps' % steps, fontsize=fs+2)
    ax.xaxis.set_label_position('top')
    plt.ylabel('Ensemble initial condition (%d cells per basin)' % ensemble, fontsize=fs+2)
    # show all ticks
    ax.set_xticks(np.arange(grid_data.shape[1]))
    ax.set_yticks(np.arange(grid_data.shape[0]))
    # label them with the respective list entries.
    assert k == 1  # TODO col labels as string types + k mixed etc
    ax.set_xticklabels(CELLTYPE_LABELS + ['mixed'], fontsize=fs)
    ax.set_yticklabels(CELLTYPE_LABELS, fontsize=fs)
    # Rotate the tick labels and set their alignment.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")

    # add gridlines
    ax.set_xticks(np.arange(-.5, grid_data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_data.shape[0], 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=1)  # grey good to split, white looks nice though

    plt.savefig(RUNS_FOLDER + os.sep + 'plot_basin_grid.pdf', dpi=100, bbox_inches='tight')
    return plt.gca()


if __name__ == '__main__':
    flag_gen_basin_grid = False
    flag_plot_basin_grid_data = True

    if flag_gen_basin_grid:
        # TODO: store run settings
        # TODO: find way to prevent reloading the interaction info from singlcell_simsetup
        ensemble = 16
        timesteps = 20
        num_proc = cpu_count() / 2
        plot = True
        t0 = time.time()
        basin_grid = gen_basin_grid(ensemble, num_proc, num_steps=timesteps, plot=plot)
        print "GRID TIMER:", time.time() - t0

    # direct data plotting
    if flag_plot_basin_grid_data:
        filestr_data = RUNS_FOLDER + os.sep + 'gen_basin_grid.txt'
        basin_grid_data = load_basin_grid(filestr_data)
        ensemble = 960
        num_steps = 100
        plot_basin_grid(basin_grid_data, ensemble, num_steps, k=1)
