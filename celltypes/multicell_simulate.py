import numpy as np
import os
import random
import matplotlib.pyplot as plt

from multicell_constants import GRIDSIZE, SEARCH_RADIUS_CELL, NUM_LATTICE_STEPS, VALID_BUILDSTRINGS, VALID_FIELDSTRINGS, FIELDSTRING, BUILDSTRING, LATTICE_PLOT_PERIOD, FIELD_REMOVE_RATIO
from multicell_lattice import build_lattice_main, get_cell_locations, prep_lattice_data_dict, write_state_all_cells
from multicell_visualize import lattice_plotter
from singlecell.singlecell_constants import FIELD_STRENGTH
from singlecell.singlecell_data_io import run_subdir_setup
from singlecell.singlecell_simsetup import XI, CELLTYPE_ID, CELLTYPE_LABELS


def run_sim(lattice, num_lattice_steps, data_dict, fieldstring=FIELDSTRING, field_remove_ratio=0.0, field_strength=FIELD_STRENGTH, plot_period=LATTICE_PLOT_PERIOD):
    """
    Form of data_dict:
        {'memory_proj_arr':
            {memory_idx: np array [N x num_steps] of projection each grid cell onto memory idx}
         'params': TODO
        }
    Notes:
        -can replace update_with_signal_field with update_state to simulate ensemble of non-intxn n**2 cells
    """
    current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup()
    n = len(lattice)
    assert n == len(lattice[0])  # work with square lattice for simplicity
    cell_locations = get_cell_locations(lattice, n)
    loc_to_idx = {pair: idx for idx, pair in enumerate(cell_locations)}
    memory_idx_list = data_dict['memory_proj_arr'].keys()

    # plot initial state of the lattice
    for mem_idx in memory_idx_list:
        lattice_plotter(lattice, 0, n, plot_lattice_folder, mem_idx)
    # get data for initial state of the lattice
    for loc in cell_locations:
        for mem_idx in memory_idx_list:
            proj = lattice[loc[0]][loc[1]].get_memories_projection()
            data_dict['memory_proj_arr'][mem_idx][loc_to_idx[loc], 0] = proj[mem_idx]

    for turn in xrange(1, num_lattice_steps + 1):
        print 'Turn ', turn
        random.shuffle(cell_locations)
        for idx, loc in enumerate(cell_locations):
            cell = lattice[loc[0]][loc[1]]
            cell.update_with_signal_field(lattice, SEARCH_RADIUS_CELL, n, fieldstring=fieldstring, ratio_to_remove=field_remove_ratio, field_strength=field_strength)
            proj = cell.get_memories_projection()
            for mem_idx in memory_idx_list:
                data_dict['memory_proj_arr'][mem_idx][loc_to_idx[loc], turn] = proj[mem_idx]
            if turn % (10*plot_period) == 0:  # plot proj visualization of each cell (takes a while; every k lat plots)
                fig, ax, proj = cell.plot_projection(use_radar=False, pltdir=plot_lattice_folder)
        if turn % plot_period == 0:  # plot the lattice
            for mem_idx in memory_idx_list:
                lattice_plotter(lattice, turn, n, plot_lattice_folder, mem_idx)

    return lattice, data_dict, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder


def main(gridize=GRIDSIZE, num_steps=NUM_LATTICE_STEPS, buildstring=BUILDSTRING, fieldstring=FIELDSTRING,
         field_remove_ratio=FIELD_REMOVE_RATIO, field_strength=FIELD_STRENGTH, plot_period=LATTICE_PLOT_PERIOD):

    # check args
    assert type(gridize) is int
    assert type(num_steps) is int
    assert type(plot_period) is int
    assert buildstring in VALID_BUILDSTRINGS
    assert fieldstring in VALID_FIELDSTRINGS
    assert 0.0 <= field_remove_ratio < 1.0
    assert 0.0 <= field_strength < 10.0

    # setup lattice IC
    type_1_idx = 5
    type_2_idx = 24
    if buildstring == "mono":
        list_of_type_idx = [type_1_idx]
    if buildstring == "dual":
        list_of_type_idx = [type_1_idx, type_2_idx]
    lattice = build_lattice_main(gridize, list_of_type_idx, buildstring)
    print list_of_type_idx

    # prep data dictionary
    data_dict = {}  # TODO: can also store params in data dict for main/run_sim then save to file
    data_dict = prep_lattice_data_dict(gridize, num_steps, list_of_type_idx, buildstring, data_dict)

    # run the simulation
    lattice, data_dict, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = \
        run_sim(lattice, num_steps, data_dict, fieldstring=fieldstring, field_remove_ratio=field_remove_ratio, field_strength=field_strength, plot_period=plot_period)

    # check the data data
    for data_idx, memory_idx in enumerate(data_dict['memory_proj_arr'].keys()):
        print data_dict['memory_proj_arr'][memory_idx]
        plt.plot(data_dict['memory_proj_arr'][memory_idx].T)
        plt.title('Projection of each grid cell onto memory %s vs grid timestep' % CELLTYPE_LABELS[memory_idx])
        plt.savefig(plot_data_folder + os.sep + '%s_%s_n%d_t%d_proj%d_remove%.2f_exo%.2f.png' %
                    (fieldstring, buildstring, gridize, num_steps, memory_idx, field_remove_ratio, field_strength))
        plt.clf()  #plt.show()

    # write cell state TODO: and data_dict to file
    # write cell state TODO: and data_dict to file
    write_state_all_cells(lattice, data_folder)

    print "\nMulticell simulation complete - output in %s" % current_run_folder
    return lattice, data_dict, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder


if __name__ == '__main__':
    n = 20  # global GRIDSIZE
    steps = 40  # global NUM_LATTICE_STEPS
    buildstring = "dual"  # mono/dual/
    fieldstring = "on"  # on/off/all, note e.g. 'off' means send info about 'off' genes only
    fieldprune = 0.8  # amount of field idx to randomly prune from each cell
    exo = 0.3  # global FIELD_STRENGTH
    plot_period=2
    main(gridize=n, num_steps=steps, buildstring=buildstring, fieldstring=fieldstring, field_remove_ratio=fieldprune,
         field_strength=exo, plot_period=plot_period)
