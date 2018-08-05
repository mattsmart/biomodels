import numpy as np
import os
import random
import matplotlib.pyplot as plt

from multicell_constants import GRIDSIZE, SEARCH_RADIUS_CELL, NUM_LATTICE_STEPS, VALID_BUILDSTRINGS, VALID_EXOSOME_STRINGS, EXOSTRING, BUILDSTRING, LATTICE_PLOT_PERIOD, FIELD_REMOVE_RATIO
from multicell_lattice import build_lattice_main, get_cell_locations, prep_lattice_data_dict, write_state_all_cells
from multicell_visualize import lattice_uniplotter, reference_overlap_plotter, lattice_projection_composite
from singlecell.singlecell_constants import EXT_FIELD_STRENGTH, APP_FIELD_STRENGTH, IPSC_CORE_GENES, IPSC_EXTENDED_GENES
from singlecell.singlecell_data_io import run_subdir_setup, runinfo_append
from singlecell.singlecell_functions import construct_app_field_from_genes
from singlecell.singlecell_simsetup import N, P, XI, CELLTYPE_ID, CELLTYPE_LABELS


def run_sim(lattice, num_lattice_steps, data_dict, exosome_string=EXOSTRING, field_remove_ratio=0.0,
            ext_field_strength=EXT_FIELD_STRENGTH, app_field=None, app_field_strength=APP_FIELD_STRENGTH,
            plot_period=LATTICE_PLOT_PERIOD, flag_uniplots=True):
    """
    Form of data_dict:
        {'memory_proj_arr':
            {memory_idx: np array [N x num_steps] of projection each grid cell onto memory idx}
         'params': TODO
        }
    Notes:
        -can replace update_with_signal_field with update_state to simulate ensemble of non-intxn n**2 cells
    """

    # Input checks
    n = len(lattice)
    assert n == len(lattice[0])  # work with square lattice for simplicity
    if app_field is not None:
        assert len(app_field) == N
        assert len(app_field[0]) == num_lattice_steps
    else:
        app_field_timestep = None

    io_dict = run_subdir_setup()
    cell_locations = get_cell_locations(lattice, n)
    loc_to_idx = {pair: idx for idx, pair in enumerate(cell_locations)}
    memory_idx_list = data_dict['memory_proj_arr'].keys()

    # plot initial state of the lattice
    if flag_uniplots:
        for mem_idx in memory_idx_list:
            lattice_uniplotter(lattice, 0, n, io_dict['latticedir'], mem_idx)
    # get data for initial state of the lattice
    for loc in cell_locations:
        for mem_idx in memory_idx_list:
            proj = lattice[loc[0]][loc[1]].get_memories_projection()
            data_dict['memory_proj_arr'][mem_idx][loc_to_idx[loc], 0] = proj[mem_idx]
    # initial condition plot
    lattice_projection_composite(lattice, 0, n, io_dict['latticedir'])
    reference_overlap_plotter(lattice, 0, n, io_dict['latticedir'])
    if flag_uniplots:
        for mem_idx in memory_idx_list:
            lattice_uniplotter(lattice, 0, n, io_dict['latticedir'], mem_idx)


    for turn in xrange(1, num_lattice_steps):
        print 'Turn ', turn
        random.shuffle(cell_locations)
        for idx, loc in enumerate(cell_locations):
            cell = lattice[loc[0]][loc[1]]
            cell.update_with_signal_field(lattice, SEARCH_RADIUS_CELL, n, exosome_string=exosome_string, ratio_to_remove=field_remove_ratio,
                                          ext_field_strength=ext_field_strength, app_field=app_field[:,turn], app_field_strength=app_field_strength)
            proj = cell.get_memories_projection()
            for mem_idx in memory_idx_list:
                data_dict['memory_proj_arr'][mem_idx][loc_to_idx[loc], turn] = proj[mem_idx]
            if turn % (40*plot_period) == 0:  # plot proj visualization of each cell (takes a while; every k lat plots)
                fig, ax, proj = cell.plot_projection(use_radar=False, pltdir=io_dict['latticedir'])
        if turn % plot_period == 0:  # plot the lattice
            lattice_projection_composite(lattice, turn, n, io_dict['latticedir'])
            reference_overlap_plotter(lattice, turn, n, io_dict['latticedir'])
            if flag_uniplots:
                for mem_idx in memory_idx_list:
                    lattice_uniplotter(lattice, turn, n, io_dict['latticedir'], mem_idx)

    return lattice, data_dict, io_dict


def main(gridsize=GRIDSIZE, num_steps=NUM_LATTICE_STEPS, buildstring=BUILDSTRING, exosome_string=EXOSTRING,
         field_remove_ratio=FIELD_REMOVE_RATIO, ext_field_strength=EXT_FIELD_STRENGTH, app_field=None,
         app_field_strength=APP_FIELD_STRENGTH, plot_period=LATTICE_PLOT_PERIOD):

    # check args
    assert type(gridsize) is int
    assert type(num_steps) is int
    assert type(plot_period) is int
    assert buildstring in VALID_BUILDSTRINGS
    assert exosome_string in VALID_EXOSOME_STRINGS
    assert 0.0 <= field_remove_ratio < 1.0
    assert 0.0 <= ext_field_strength < 10.0

    # setup lattice IC
    type_1_idx = 5
    type_2_idx = 36
    flag_uniplots = True
    if buildstring == "mono":
        list_of_type_idx = [type_1_idx]
    if buildstring == "dual":
        list_of_type_idx = [type_1_idx, type_2_idx]
    if buildstring == "memory_sequence":
        flag_uniplots = False
        list_of_type_idx = range(P)
        random.shuffle(list_of_type_idx)  # shuffle or not?
    lattice = build_lattice_main(gridsize, list_of_type_idx, buildstring)
    #print list_of_type_idx

    # prep data dictionary
    data_dict = {}  # TODO: can also store params in data dict for main/run_sim then save to file
    data_dict = prep_lattice_data_dict(gridsize, num_steps, list_of_type_idx, buildstring, data_dict)

    # run the simulation
    lattice, data_dict, io_dict = \
        run_sim(lattice, num_steps, data_dict, exosome_string=exosome_string, field_remove_ratio=field_remove_ratio,
                ext_field_strength=ext_field_strength, app_field=app_field, app_field_strength=app_field_strength,
                plot_period=plot_period, flag_uniplots=flag_uniplots)

    # check the data dict
    for data_idx, memory_idx in enumerate(data_dict['memory_proj_arr'].keys()):
        print data_dict['memory_proj_arr'][memory_idx]
        plt.plot(data_dict['memory_proj_arr'][memory_idx].T)
        plt.ylabel('Projection of all cells onto type: %s' % CELLTYPE_LABELS[memory_idx])
        plt.xlabel('Time (full lattice steps)')
        plt.savefig(io_dict['plotdir'] + os.sep + '%s_%s_n%d_t%d_proj%d_remove%.2f_exo%.2f.png' %
                    (exosome_string, buildstring, gridsize, num_steps, memory_idx, field_remove_ratio, ext_field_strength))
        plt.clf()  #plt.show()

    # write cell state TODO: and data_dict to file
    write_state_all_cells(lattice, io_dict['datadir'])

    print "\nMulticell simulation complete - output in %s" % io_dict['baseedir']
    return lattice, data_dict, io_dict


if __name__ == '__main__':
    n = 6  # global GRIDSIZE
    steps = 20  # global NUM_LATTICE_STEPS
    buildstring = "dual"  # mono/dual/memory_sequence/
    fieldstring = "on"  # on/off/all, note e.g. 'off' means send info about 'off' genes only
    fieldprune = 0.8  # amount of external field idx to randomly prune from each cell
    ext_field_strength = 0.3                                                  # global EXT_FIELD_STRENGTH
    app_field = construct_app_field_from_genes(IPSC_EXTENDED_GENES, steps)        # size N x timesteps or None
    app_field_strength = 0.0 #100.0                                                  # global APP_FIELD_STRENGTH
    plot_period = 4
    main(gridsize=n, num_steps=steps, buildstring=buildstring, exosome_string=fieldstring, field_remove_ratio=fieldprune,
         ext_field_strength=ext_field_strength, app_field=app_field, app_field_strength=app_field_strength, plot_period=plot_period)
