import csv
import datetime
import numpy as np
import os
import random
import sys
import time
from math import ceil, floor
from numpy.random import randint

from multicell_class import SpatialCell
from singlecell.singlecell_data_io import run_subdir_setup
from singlecell.singlecell_simsetup import XI, CELLTYPE_ID, CELLTYPE_LABELS
from utils import make_video


# Constants
# =================================================
# simulation lattice parameters
n = 4
search_radius_cell = 1
assert search_radius_cell < n / 2

"""
standard_run_time = 1 * 24.0  # typical simulation time in h
turn_rate = 10.0  # 2.0  # average turns between each division; simulation step size
time_per_turn = min(donor_A_div_mean, donor_B_div_mean) / turn_rate  # currently 20min / turn rate
plots_period_in_turns = 5  # 1 or 1000 or 2 * turn_rate
total_turns = int(ceil(standard_run_time / time_per_turn))

# miscellaneous simulation settings
video_flag = False  # WARNING: auto video creation requires proper ffmpeg setup and folder permissions and luck
FPS = 6
"""


# Functions
# =================================================
def printer(lattice):
    for i in xrange(n):
        str_lst = [lattice[i][j].label for j in xrange(n)]
        print " " + ' '.join(str_lst)
    print


def build_lattice_default(n):
    lattice = [[0 for y in xrange(n)] for x in xrange(n)]  # TODO: this can be made faster as np array
    for i in xrange(n):
        for j in xrange(n):
            #celltype = np.random.choice(CELLTYPE_LABELS)
            celltype = CELLTYPE_LABELS[5]
            init_state = XI[:, CELLTYPE_ID[celltype]]
            lattice[i][j] = SpatialCell(init_state, "%d,%d_%s" % (i,j,celltype), [i,j])
    return lattice


def get_label(lattice, loc):
    return lattice[loc[0]][loc[1]].label


def get_cell_locations(lattice):
    cell_locations = []
    for i in xrange(n):
        for j in xrange(n):
            loc = (i, j)
            if isinstance(lattice[i][j], SpatialCell):
                cell_locations.append(loc)
    return cell_locations


def write_state_all_cells(lattice, data_folder):
    print "Writing states to file.."
    for i in xrange(len(lattice)):
        for j in xrange(len(lattice[0])):
            lattice[i][j].write_state(data_folder)
    print "Done"


def run_sim(lattice, duration):

    """
    # get stats for lattice initial condition before entering simulation loop, add to lattice data
    print 'Turn ', 0, ' : Time Elapsed ', 0.0, "h"
    dict_counts = count_cells()
    lattice_data[0, :] = np.array([0, 0.0, dict_counts['_'], dict_counts['D_a'], dict_counts['D_b'], dict_counts['B']])

    # plot initial conditions
    lattice_plotter(lattice, 0.0, n, dict_counts, plot_lattice_folder)  # TODO Re-add

    # simulation loop initialization
    new_cell_locations = []
    locations_to_remove = []
    cell_locations = get_cell_locations()
    """

    cell_locations = get_cell_locations(lattice)

    proj_data = np.zeros((16, duration+1))
    proj_data[:,0] = np.ones(16)
    loc_to_idx = {pair: idx for idx, pair in enumerate(cell_locations)}


    current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup()
    n = len(lattice)
    assert n == len(lattice[0])

    for turn in xrange(1, duration + 1):
        print '\nTurn ', turn
        random.shuffle(cell_locations)  # TODO USE AND TEST SWAPPED RANDOM INITIALIZER
        for idx, loc in enumerate(cell_locations):
            cell = lattice[loc[0]][loc[1]]
            #print "BEFORE", idx, loc, cell, sum(np.abs(cell.state)), len(cell.state), set(cell.state)
            cell.update_with_signal_field(lattice, search_radius_cell, n)
            #print "AFTER", idx, loc, cell, sum(np.abs(cell.state)), len(cell.state), set(cell.state)
            #cell.plot_projection(pltdir=plot_lattice_folder)

            #proj = cell.get_memories_projection()
            #proj_data[loc_to_idx[loc],turn] = proj[7]

        """
        # periodically plot the lattice (it takes a while)
        if turn % plots_period_in_turns == 0:
            t0_a = time.clock()
            t0_b = time.time()
            lattice_plotter(lattice, turn * time_per_turn, n, dict_counts, plot_lattice_folder)
            print "PLOT process time:", time.clock() - t0_a
            print "PLOT wall time:", time.time() - t0_b
        """
    #print proj_data
    import matplotlib.pyplot as plt
    #plt.plot(proj_data.T)
    #plt.show()
    return lattice, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder


# Main Function
# =================================================
def main(lattice_size, duration):
    # choose ICs
    lattice = build_lattice_default(lattice_size)

    # run the simulation
    lattice, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_sim(lattice, duration)

    # write data to file
    write_state_all_cells(lattice, data_folder)

    print "\nDone Multicell Sim"
    return

if __name__ == '__main__':
    print XI[:, 5], set(XI[:, 5])  # used 7 first, fix above too
    duration = 100
    main(n, duration)
