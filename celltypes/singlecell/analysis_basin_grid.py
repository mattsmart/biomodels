import matplotlib.pyplot as plt
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count

from analysis_basin_transitions import ensemble_projection_timeseries, fast_basin_stats, get_init_info, OCC_THRESHOLD, ANNEAL_BETA
from singlecell_class import Cell
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
    basin_grid = np.zeros((len(CELLTYPE_LABELS), len(CELLTYPE_LABELS)+k))
    for idx, celltype in enumerate(CELLTYPE_LABELS):
        if saveall:
            proj_timeseries_array, basin_occupancy_timeseries = \
                ensemble_projection_timeseries(init_cond, ensemble, num_proc, num_steps=num_steps, beta=beta,
                                               occ_threshold=occ_threshold, plot=True, anneal=True)
        else:
            init_state, init_id = get_init_info(celltype)
            endpoint_dict, transfer_dict, proj_timeseries_array, basin_occupancy_timeseries = \
                fast_basin_stats(celltype, init_state, init_id, ensemble, num_processes,
                                 num_steps=num_steps, beta=beta, anneal=True, verbose=False, occ_threshold=occ_threshold)
        # fill in row of grid data from eah celltype simulation
        basin_grid[idx, :] = basin_occupancy_timeseries[:,-1]
    if save:
        np.savetxt(RUNS_FOLDER + os.sep + 'gen_basin_grid.txt', basin_grid, delimiter=',')
    if plot:
        plot_basin_grid(basin_grid, ensemble, num_steps, k=k)
    return basin_grid


def load_basin_grid(filestr_data):
    # TODO: prepare IO functions for standardized sim settings dict struct
    basin_grid = np.loadtxt(filestr_data, delimiter=',', dtype=float)
    #sim_settings = load_sim_settings(filestr_settings)
    return basin_grid


def plot_basin_grid(grid_data, ensemble, steps, k=1, ax=None):
    """
    plot matrix G_ij of size p x (p + k): grid of data between 0 and 1
    each row represents one of the p encoded basins as an initial condition
    each column represents an endpoint of the simulation starting at a given basin (row)
    G_ij would represent: starting in cell type i, G_ij of the ensemble transitioned to cell type j
    k represents the number of extra tracked states, by default this is 1 (i.e. mixed state, not in any basin)
    """
    assert grid_data.shape == (len(CELLTYPE_LABELS), len(CELLTYPE_LABELS) + k)

    if not ax:
        ax = plt.gca()
    # plot the heatmap
    imshow_kw = {'cmap': 'YlGnBu', 'vmin': 0.0, 'vmax': 1.0}
    im = ax.imshow(grid_data, **imshow_kw)
    # create colorbar
    cbar_kw = {}
    cbarlabel = 'Basin occupancy fraction'
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    plt.title('Basin grid transition data (%d cells, %d steps)' % (ensemble, steps))
    plt.ylabel(CELLTYPE_LABELS)
    # TODO col labels as string types + k mixed etc
    assert k == 1
    plt.xlabel(CELLTYPE_LABELS + ['mixed'])
    # Rotate the tick labels and set their alignment.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    plt.savefig(RUNS_FOLDER + os.sep + 'plot_basin_grid.png')
    return plt.gca()


if __name__ == '__main__':
    flag_gen_basin_grid = False
    flag_plot_basin_grid_data = True

    if flag_gen_basin_grid:
        # TODO: store run settings
        ensemble = 10
        timesteps = 10
        num_proc = cpu_count() / 2
        plot = True
        basin_grid = gen_basin_grid(ensemble, num_proc, num_steps=timesteps, plot=plot)

    # direct data plotting
    if flag_plot_basin_grid_data:
        filestr_data = RUNS_FOLDER + os.sep + 'gen_basin_grid.txt'
        basin_grid_data = load_basin_grid(filestr_data)
        ensemble = 0
        num_steps = 0
        plot_basin_grid(basin_grid_data, ensemble, num_steps, k=1)
