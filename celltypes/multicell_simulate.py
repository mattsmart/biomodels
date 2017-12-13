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

# CROSS-PACKAGE IMPORT HACK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir)))
from utils import make_video


# Constants
# =================================================
# simulation dimensions
n = 100  # up to 1000 tested as feasible

# simulation lattice parameters
search_radius_cell = 1
assert search_radius_cell < n / 2

# simulation time settings
standard_run_time = 1 * 24.0  # typical simulation time in h
turn_rate = 10.0  # 2.0  # average turns between each division; simulation step size
time_per_turn = min(donor_A_div_mean, donor_B_div_mean) / turn_rate  # currently 20min / turn rate
plots_period_in_turns = 5  # 1 or 1000 or 2 * turn_rate
total_turns = int(ceil(standard_run_time / time_per_turn))

# miscellaneous simulation settings
video_flag = False  # WARNING: auto video creation requires proper ffmpeg setup and folder permissions and luck
FPS = 6


# Initiate Cell Lattice and Data Directory
# =================================================
lattice = [[Empty([x, y]) for y in xrange(n)] for x in xrange(n)]  # this can be made faster as np array
lattice_data = np.zeros((total_turns + 1, 6))  # sublists are [turn, time, E, D_a, D_b, B]


# Functions
# =================================================
def printer():
    for i in xrange(n):
        str_lst = [lattice[i][j].label for j in xrange(n)]
        print " " + ' '.join(str_lst)
    print


def build_lattice_colonies():
    pivot = int(0.45 * n)
    anti_pivot = n - pivot - 1
    lattice[pivot][pivot] = DonorTypeA([pivot, pivot], time_to_div_distribution='uniform')
    lattice[anti_pivot][anti_pivot] = DonorTypeB([anti_pivot, anti_pivot], time_to_div_distribution='uniform')
    return lattice


def build_lattice_random(seed=5):
    # seed: determines ratio of donors to recipients for random homogeneous conditions
    random_lattice = randint(seed, size=(n, n))
    for i in xrange(n):
        for j in xrange(n):
            m = random_lattice[i][j]
            if m == 0:
                lattice[i][j] = DonorTypeA([i, j], time_to_div_distribution='uniform')
            elif m == 1:
                lattice[i][j] = DonorTypeB([i, j], time_to_div_distribution='uniform')
            elif m in range(2, seed):
                lattice[i][j] = Empty([i, j])
    print random_lattice, "\n"
    return


def build_lattice_diag():
    for i in xrange(n):
        for j in xrange(n):
            if j < i:
                lattice[i][j] = DonorTypeA([i, j], time_to_div_distribution='uniform')
            else:
                lattice[i][j] = DonorTypeB([i, j], time_to_div_distribution='uniform')
    return


def build_lattice_concentric_random(sprinkle=0.2):
    """Two concentric circles with potential sprinkling (sparse placement) of each bacterial type
    Args:
        sprinkle_prob: default 0.2, 1.0 is completely filled circles
    """
    assert 0.0 <= sprinkle <= 1.0
    radius_inner = np.ceil(n * 0.10)
    radius_outer = np.ceil(n * 0.20)
    for i in xrange(n):
        for j in xrange(n):
            radius = np.sqrt((i - n/2)**2 + (j - n/2)**2)
            # probability module
            m = randint(0, 100)
            if 100*sprinkle >= m:
                insert_cell = True
            else:
                insert_cell = False
            # circle module
            if radius <= radius_inner and insert_cell:
                lattice[i][j] = DonorTypeA([i, j], time_to_div_distribution='uniform')
            elif radius <= radius_outer and insert_cell:
                lattice[i][j] = DonorTypeB([i, j], time_to_div_distribution='uniform')
            else:
                continue
    return


def build_lattice_sprinkle(ratio_a=0.2, ratio_b=0.2, ic_radius_fraction=0.1):
    """Sprinkle cells in the center, IC edge size n * m
    Args:
        ratio_a, ratio_b: determine sprinkling in IC region
        ic_radius_fraction: size fraction of IC radius
    """
    # ic setup
    ic_radius_units = int(n * ic_radius_fraction)
    top_left = int(n / 2 - ic_radius_units)
    # probability setup
    sample_range_a_upper = int(100 * ratio_a)
    sample_range_b_upper = int(100 * (ratio_a + ratio_b))
    random_lattice = randint(100, size=(2 * ic_radius_units, 2 * ic_radius_units))
    # generate ic
    for idx_i, lattice_i in enumerate(xrange(top_left, top_left + 2 * ic_radius_units)):
        for idx_j, lattice_j in enumerate(xrange(top_left, top_left + 2 * ic_radius_units)):
            m = random_lattice[idx_i][idx_j]
            if 0 <= m <= sample_range_a_upper:
                lattice[lattice_i][lattice_j] = DonorTypeA([lattice_i, lattice_j], time_to_div_distribution='uniform')
            elif sample_range_a_upper <= m <= sample_range_b_upper:
                lattice[lattice_i][lattice_j] = DonorTypeB([lattice_i, lattice_j], time_to_div_distribution='uniform')
            else:
                lattice[lattice_i][lattice_j] = Empty([lattice_i, lattice_j])
    print random_lattice, "\n"
    return

"""
def get_label(loc):
    return lattice[loc[0]][loc[1]].label


def divide(cell, new_cell_locations, dict_counts):
    success = 0
    if cell.time_to_div < 0:
        empty_neighbours = cell.get_label_surroundings('_', search_radius_cell)
        if len(empty_neighbours) > 0:
            success = 1
            daughter_loc = random.choice(empty_neighbours)
            if 'D_a' == cell.label:
                lattice[daughter_loc[0]][daughter_loc[1]] = DonorTypeA(daughter_loc)
            elif 'D_b' == cell.label:
                lattice[daughter_loc[0]][daughter_loc[1]] = DonorTypeB(daughter_loc)
            else:
                raise Exception("Illegal cell type")
            # update parent cell time to div
            cell.set_normal_time_to_div()
            # update tracking variables
            new_cell_locations.append(daughter_loc)
            dict_counts[cell.label] += 1
            dict_counts['_'] -= 1
    return success
"""

def run_sim():

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

    for turn in xrange(1, total_turns + 1):
        print '\nTurn ', turn, ' : Time Elapsed ', turn * time_per_turn, "h"

        # update tracking lists
        for loc in locations_to_remove:
            cell_locations.remove(loc)
        cell_locations = cell_locations + new_cell_locations
        random.shuffle(cell_locations)  # TODO USE AND TEST SWAPPED RANDOM INITIALIZER

        # reset utility lists
        new_cell_locations = []
        locations_to_remove = []

        # timestep profiling
        t0_a = time.clock()
        t0_b = time.time()

        for loc in cell_locations:
            cell = lattice[loc[0]][loc[1]]

            # debris behaviour
            if is_debris(loc):
                cell.time_to_decay -= time_per_turn
                if cell.time_to_decay < 0:
                    lattice[loc[0]][loc[1]] = Empty(loc)
                    locations_to_remove.append(loc)
                    dict_counts['B'] -= 1
                    dict_counts['_'] += 1

            # donor behaviour
            elif is_donor_type_a(loc) or is_donor_type_b(loc):
                # death check for poisoned cells first
                death_flag = cell.decrement_poison_timer_and_report_death()
                if death_flag:
                    lattice[loc[0]][loc[1]] = Debris(loc)
                    dict_counts['B'] += 1
                    dict_counts[cell.label] -= 1
                else:
                    # decrement division timer
                    cell.time_to_div -= time_per_turn
                    # flip coin to see if shooting or dividing first
                    coinflip_heads = randint(2)
                    if coinflip_heads:
                        divide(cell, new_cell_locations, dict_counts)
                        shoot(cell, dict_counts)
                    else:
                        shoot(cell, dict_counts)
                        divide(cell, new_cell_locations, dict_counts)

            else:
                print "WARNING - Cell not D_a or D_b or B"
                print "Cell label and location:", cell.label, cell.location
                raise Exception("Cell not D_a or D_b or B")

        # get lattice stats for this timestep
        counts = count_cells()
        lattice_data[turn, :] = np.array([turn, turn * time_per_turn, dict_counts['_'], dict_counts['D_a'], dict_counts['D_b'], dict_counts['B']])
        # print "COUNTS actual", counts
        # print "COUNTS dict", dict_counts

        # timestep profiling
        print "SIM process time:", time.clock() - t0_a
        print "SIM wall time:", time.time() - t0_b

        # periodically plot the lattice (it takes a while)
        if turn % plots_period_in_turns == 0:
            t0_a = time.clock()
            t0_b = time.time()
            lattice_plotter(lattice, turn * time_per_turn, n, dict_counts, plot_lattice_folder)
            print "PLOT process time:", time.clock() - t0_a
            print "PLOT wall time:", time.time() - t0_b

    return lattice_data


# Main Function
# =================================================
def main():
    # choose ICs
    #build_lattice_random()
    #build_lattice_colonies()
    #build_lattice_diag()
    #build_lattice_concentric_random()
    build_lattice_sprinkle()

    # run the simulation
    run_sim()

    # write data to file
    """
    data_name = "lattice_data.csv"
    data_file = os.path.join(data_folder, data_name)
    with open(data_file, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(lattice_data)

    # convert lattice_data to a dictionary and plot it
    data_dict = {'iters': lattice_data[:, 0],
                 'time': lattice_data[:, 1],
                 'E': lattice_data[:, 2],
                 'D_a': lattice_data[:, 3],
                 'D_b': lattice_data[:, 4],
                 'B': lattice_data[:, 5]}
    data_plotter(data_dict, data_file, plot_data_folder)
    
    # create video of results
    if video_flag:
        video_path = os.path.join(current_run_folder, "plot_lattice_%dh_%dfps.mp4" % (standard_run_time, FPS))
        make_video.make_video_ffmpeg(plot_lattice_folder, video_path, fps=FPS)
    """

    print "\nDone!"
    return

if __name__ == '__main__':
    main()
