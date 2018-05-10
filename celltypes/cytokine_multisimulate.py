import numpy as np
import os
import random
import matplotlib.pyplot as plt

from singlecell.cytokine_settings import APP_FIELD_STRENGTH
from cytokine_lattice import build_cytokine_lattice_mono
from multicell_lattice import get_cell_locations, write_state_all_cells


GRIDSIZE = 5
NUM_LATTICE_STEPS = 5
SEARCH_RADIUS_CELL = 1  # TODO find nice way to have none flag here for inf range singles / homogeneous?


def run_cytokine_network(lattice, num_lattice_steps, app_field=None, app_field_strength=APP_FIELD_STRENGTH, flag_write=False):

    # Input checks
    n = len(lattice)
    assert n == len(lattice[0])  # work with square lattice for simplicity
    if app_field is not None:
        assert len(app_field) == N
        assert len(app_field[0]) == num_lattice_steps
    else:
        app_field_timestep = None

    # io
    if flag_write:
        current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup(run_subfolder=RUNS_SUBDIR_CYTOKINES)
        dirs = [current_run_folder, data_folder, plot_lattice_folder, plot_data_folder]
    else:
        dirs = None

    cell_locations = get_cell_locations(lattice, n)
    loc_to_idx = {pair: idx for idx, pair in enumerate(cell_locations)}
    memory_idx_list = data_dict['memory_proj_arr'].keys()

    # plot initial state of the lattice
    """
    if flag_uniplots:
        for mem_idx in memory_idx_list:
            lattice_uniplotter(lattice, 0, n, plot_lattice_folder, mem_idx)
    """

    # initial condition plot
    """
    lattice_projection_composite(lattice, 0, n, plot_lattice_folder)
    reference_overlap_plotter(lattice, 0, n, plot_lattice_folder)
    if flag_uniplots:
        for mem_idx in memory_idx_list:
            lattice_uniplotter(lattice, 0, n, plot_lattice_folder, mem_idx)
    """

    for turn in xrange(1, num_lattice_steps):
        print 'Turn ', turn
        random.shuffle(cell_locations)
        for idx, loc in enumerate(cell_locations):
            cell = lattice[loc[0]][loc[1]]
            cell.update_with_signal_field(lattice, SEARCH_RADIUS_CELL, n, app_field=app_field[:,turn], app_field_strength=app_field_strength)
            proj = cell.get_memories_projection()
            for mem_idx in memory_idx_list:
                data_dict['memory_proj_arr'][mem_idx][loc_to_idx[loc], turn] = proj[mem_idx]
            if turn % (40*plot_period) == 0:  # plot proj visualization of each cell (takes a while; every k lat plots)
                fig, ax, proj = cell.plot_projection(use_radar=False, pltdir=plot_lattice_folder)
        if turn % plot_period == 0:  # plot the lattice
            lattice_projection_composite(lattice, turn, n, plot_lattice_folder)
            reference_overlap_plotter(lattice, turn, n, plot_lattice_folder)
            if flag_uniplots:
                for mem_idx in memory_idx_list:
                    lattice_uniplotter(lattice, turn, n, plot_lattice_folder, mem_idx)

    return lattice, dirs


def wrapper_cytokine_network(gridsize=GRIDSIZE, num_steps=NUM_LATTICE_STEPS, app_field=None, app_field_strength=APP_FIELD_STRENGTH, flag_write=False):

    # setup lattice IC
    lattice, spin_labels, intxn_matrix, applied_field_const, init_state, signal_matrix = build_cytokine_lattice_mono(gridsize)

    # run the simulation
    lattice, dirs = run_cytokine_network(lattice, num_steps, app_field=app_field, app_field_strength=app_field_strength, flag_write=flag_write)

    # write cell state TODO: and data_dict to file
    #write_state_all_cells(lattice, data_folder)

    print "\nMulticell simulation complete - output in %s" % dirs[0]
    return lattice, dirs


if __name__ == '__main__':
    wrapper_cytokine_network()