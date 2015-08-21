import csv
import datetime
import numpy as np
import os
import random
import sys
import time
from math import ceil, floor
from numpy.random import randint

from plot_data import data_plotter
from plot_lattice import lattice_plotter

# CROSS-PACKAGE IMPORT HACK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir)))
from utils import make_video


"""
TODO
-convert lattice to np array and ref with tuples instead of separate loc 0 and loc 1 (cleaner, maybe faster)
-make paths independent of OS (use os.path.join(...))

SPEED
-instead of explicit class structure for the states, could just use dicts (should be faster)
-use location tuples instead of lists (faster assigning)
-all to numpy arrays
-store cell type as well as position for faster referencing?

IMPORTANT
- turn time needs to be tied to the time to shoot/kill -- fix this
- make donorA and donorB inherit from a general donor class -- this WILL reduce bugs lol
"""


# IO
# =================================================
runs_folder = "runs" + os.sep  # store timestamped runs here
current_time = datetime.datetime.now().strftime("%Y-%m-%d %I.%M.%S%p")
time_folder = current_time + os.sep
current_run_folder = runs_folder + time_folder

# subfolders in the timestamped run directory:
data_folder = os.path.join(current_run_folder, "data")
plot_lattice_folder = os.path.join(current_run_folder, "plot_lattice")
plot_data_folder = os.path.join(current_run_folder, "plot_data")

dir_list = [runs_folder, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder]
for dirs in dir_list:
    if not os.path.exists(dirs):
        os.makedirs(dirs)

# Constants
# =================================================
# simulation dimensions
n = 1000  # up to 1000 tested as feasible

# simulation lattice parameters
search_radius_bacteria = 1
assert search_radius_bacteria < n / 2

# cholera-specific parameters
div_mean_cholera = 20.0 / 60.0  # 20 minutes in hours
div_sd_cholera = 2.5 / 60.0  # 2.5 minutes in hours
death_by_poison_mean_cholera = 5.0 / 60.0  # 5 minutes in hours
targets_min_cholera = 0
targets_max_cholera = 5

# donor type A - division and target timings
donor_A_div_mean = div_mean_cholera
donor_A_div_sd = div_sd_cholera
donor_A_targets_min = targets_min_cholera
donor_A_targets_max = targets_max_cholera
donor_A_death_by_poison_mean = death_by_poison_mean_cholera

# donor type B - division and target timings
donor_B_div_mean = div_mean_cholera
donor_B_div_sd = div_sd_cholera
donor_B_targets_min = targets_min_cholera
donor_B_targets_max = targets_max_cholera
donor_B_death_by_poison_mean = death_by_poison_mean_cholera

# miscellaneous cell settings
#debris_decay_time = div_mean_cholera * 2.01
debris_decay_time = div_mean_cholera * 2.0
debris_decay_sd = 5.0

# simulation time settings
standard_run_time = 1 * 24.0  # typical simulation time in h
turn_rate = 10.0  # 2.0  # average turns between each division; simulation step size
time_per_turn = min(donor_A_div_mean, donor_B_div_mean) / turn_rate  # currently 20min / turn rate
plots_period_in_turns = 10  # 1 or 1000 or 2 * turn_rate
total_turns = int(ceil(standard_run_time / time_per_turn))

# miscellaneous simulation settings
video_flag = False  # WARNING: auto video creation requires proper ffmpeg setup and folder permissions and luck
FPS = 6


# Classes
# =================================================
# represents the state of a lattice cell: empty, donor, recipient
class Cell(object):
    def __init__(self, label, location):
        self.label = label  # symbol; either "(_) Empty, (R)eceiver, (D)onor"
        self.location = location  # list [i,j]

    def __str__(self):
        return self.label

    def get_surroundings_square(self, search_radius):
        """Specifies the location of the top left corner of the search square
        Args:
            search_radius: half-edge length of the square
        Returns:
            list of locations; length should be (2 * search_radius + 1) ** 2 (- 1 remove self?)
        Notes:
            - periodic BCs apply, so search boxes wrap around at boundaries
            - note that we assert that search_radius be less than half the grid size
            - may have different search radius depending om context (neighbouring bacteria / empty cells)
            - currently DOES NOT remove the original location
        """
        row = self.location[0]
        col = self.location[1]
        surroundings = [[row_to_search % n, col_to_search % n]
                        for row_to_search in xrange(row - search_radius, row + search_radius + 1)
                        for col_to_search in xrange(col - search_radius, col + search_radius + 1)]
        surroundings.remove(self.location)  # TODO test behaviour
        return surroundings

    def get_label_surroundings(self, cell_label, search_radius):
        if cell_label not in ['_', 'D_a', 'D_b', 'B']:
            raise Exception("Illegal cell label (_, D_a, D_b, or B)")
        neighbours = self.get_surroundings_square(search_radius=search_radius)
        neighbours_of_specified_type = []
        for loc in neighbours:  # TODO should skip self when going through (i.e. don't count your current position)
            if cell_label == lattice[loc[0]][loc[1]].label:
                neighbours_of_specified_type.append(loc)
        return neighbours_of_specified_type


class Empty(Cell):
    def __init__(self, location):
        Cell.__init__(self, '_', location)


class DonorTypeA(Cell):
    def __init__(self, location, time_to_div_distribution='normal'):
        Cell.__init__(self, 'D_a', location)
        self.div_mean = donor_A_div_mean
        self.div_sd = donor_A_div_sd
        self.targets_min = donor_A_targets_min
        self.targets_max = donor_A_targets_max
        self.time_to_div = None
        if time_to_div_distribution == 'normal':
            self.set_normal_time_to_div()
        elif time_to_div_distribution == 'uniform':
            self.set_uniform_time_to_div()
        else:
            raise Exception("distribution must be 'normal' or 'uniform'")
        self.death_by_poison_mean = donor_A_death_by_poison_mean
        self.time_to_death_by_poison = None

    def set_normal_time_to_div(self):
        self.time_to_div = np.random.normal(self.div_mean, self.div_sd)

    def set_uniform_time_to_div(self):
        self.time_to_div = np.random.uniform(0.0, self.div_mean)

    def start_poison_timer(self):
        if self.time_to_death_by_poison is None:
            self.time_to_death_by_poison = self.death_by_poison_mean

    def decrement_poison_timer_and_report_death(self):
        death_flag = 0
        if self.time_to_death_by_poison is not None:
            self.time_to_death_by_poison -= time_per_turn
            if self.time_to_death_by_poison < 0:
                death_flag = 1
        return death_flag


class DonorTypeB(Cell):
    def __init__(self, location, time_to_div_distribution='normal'):
        Cell.__init__(self, 'D_b', location)
        self.div_mean = donor_B_div_mean
        self.div_sd = donor_B_div_sd
        self.targets_min = donor_B_targets_min
        self.targets_max = donor_B_targets_max
        self.time_to_div = None
        if time_to_div_distribution == 'normal':
            self.set_normal_time_to_div()
        elif time_to_div_distribution == 'uniform':
            self.set_uniform_time_to_div()
        else:
            raise Exception("distribution must be 'normal' or 'uniform'")
        self.death_by_poison_mean = donor_B_death_by_poison_mean
        self.time_to_death_by_poison = None

    def set_normal_time_to_div(self):
        self.time_to_div = np.random.normal(self.div_mean, self.div_sd)

    def set_uniform_time_to_div(self):
        self.time_to_div = np.random.uniform(0.0, self.div_mean)

    def start_poison_timer(self):
        if self.time_to_death_by_poison is None:
            self.time_to_death_by_poison = self.death_by_poison_mean

    def decrement_poison_timer_and_report_death(self):
        death_flag = 0
        if self.time_to_death_by_poison is not None:
            self.time_to_death_by_poison -= time_per_turn
            if self.time_to_death_by_poison < 0:
                death_flag = 1
        return death_flag


class Debris(Cell):
    def __init__(self, location):
        Cell.__init__(self, 'B', location)
        #self.time_to_decay = debris_decay_time  # time (in hours) until debris decays
        self.time_to_decay = np.random.normal(debris_decay_time, debris_decay_sd)


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


def is_empty(loc):
    return '_' == lattice[loc[0]][loc[1]].label


def is_donor_type_a(loc):
    return 'D_a' == lattice[loc[0]][loc[1]].label


def is_donor_type_b(loc):
    return 'D_b' == lattice[loc[0]][loc[1]].label


def is_debris(loc):
    return 'B' == lattice[loc[0]][loc[1]].label


def get_label(loc):
    return lattice[loc[0]][loc[1]].label


def divide(cell, new_cell_locations, dict_counts):
    success = 0
    if cell.time_to_div < 0:
        empty_neighbours = cell.get_label_surroundings('_', search_radius_bacteria)
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


def shoot(cell, dict_counts):
    success = 0  # note that successful firing AND killing = 1
    surroundings = cell.get_surroundings_square(search_radius_bacteria)
    target_count = randint(cell.targets_min, cell.targets_max + 1)
    target_locations = random.sample(surroundings, target_count)
    for target_loc in target_locations:
        target = lattice[target_loc[0]][target_loc[1]]
        assert cell.label in ['D_a', 'D_b']
        if target.label in ['D_a', 'D_b'] and target.label != cell.label:  # target is valid and susceptible
            success = 1
            target.start_poison_timer()
    return success


def count_cells():  # returns a dict of current cell counts: [# of empty, # of recipient, # of donor]
    keys = ['_', 'D_a', 'D_b', 'B']
    counts = {key: 0 for key in keys}
    for i in xrange(n):
        for j in xrange(n):
            loc = (i, j)
            counts[get_label(loc)] += 1
    return counts


def get_cell_locations():
    cell_locations = []
    for i in xrange(n):
        for j in xrange(n):
            loc = (i, j)
            if not is_empty(loc):
                cell_locations.append(loc)
    return cell_locations


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

    print "\nDone!"
    return

if __name__ == '__main__':
    main()
