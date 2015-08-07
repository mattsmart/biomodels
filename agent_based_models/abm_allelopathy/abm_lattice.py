import csv
import datetime
import numpy as np
import os
import random
import time
from math import ceil, floor
from numpy.random import randint

from plot_data import data_plotter
from plot_lattice import lattice_plotter


"""
TODO
-remove or augment the maturity module
-ICs that resemble the PDEs
-convert lattice to np array and ref with tuples instead of separate loc 0 and loc 1 (cleaner, maybe faster)

SPEED
-instead of explicit class structure for the states, could just use dicts (should be faster)
-use location tuples instead of lists (faster assigning)
-faster and better probability modules
-all to numpy arrays
-store cell type as well as position for faster referencing?

PLOTTING SPEED
-could save more time by not storing or plotting empties?

IMPORTANT
-turn time needs to be tied to the time to shoot/kill
"""


# IO
# =================================================
runs_folder = "runs\\"  # store timestamped runs here
current_time = datetime.datetime.now().strftime("%Y-%m-%d %I.%M.%S%p")
time_folder = current_time + "\\"
current_run_folder = runs_folder + time_folder

# subfolders in the timestamped run directory:
data_folder = current_run_folder + "data\\"
plot_lattice_folder = current_run_folder + "plot_lattice\\"
plot_data_folder = current_run_folder + "plot_data\\"

dir_list = [runs_folder, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder]
for dirs in dir_list:
    if not os.path.exists(dirs):
        os.makedirs(dirs)


# Constants
# =================================================
# simulation dimensions
n = 100  # up to 1000 tested as feasible

# simulation lattice parameters
search_radius_bacteria = 1
assert search_radius_bacteria < n / 2

# division and recovery times
div_time_cholera = 0.333
expected_donor_A_div_time = div_time_cholera  # avg time for 1 R cell to divide in h
expected_donor_B_div_time = div_time_cholera  # avg time for 1 D cell to divide in h

# conjugation rate
expected_shoot_time = div_time_cholera / 2  # avg time for 1 cell to conjugate in h (Jama: 10h for e.coli, more for staph)
expected_donor_A_shoot_time = expected_shoot_time
expected_donor_B_shoot_time = expected_shoot_time

# simulation time settings
standard_run_time = 1 * 24.0  # typical simulation time in h
turn_rate = 10.0  # 2.0  # average turns between each division; simulation step size
time_per_turn = min(expected_donor_A_div_time, expected_donor_B_div_time) / turn_rate
plots_period_in_turns = 4 * turn_rate  # 1 or 1000 or 2 * turn_rate
total_turns = int(ceil(standard_run_time / time_per_turn))

# miscellaneous cell settings
expected_donor_A_div_refractory_turns = ceil((expected_donor_A_div_time / 2) / time_per_turn)  # refractory period after division in turns
expected_donor_B_div_refractory_turns = ceil((expected_donor_B_div_time / 2) / time_per_turn)  # refractory period after division in turns
debris_decay_time = div_time_cholera * 2.01


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
    def __init__(self, location):
        Cell.__init__(self, 'D_a', location)
        self.pause = 0  # 0 if cell is active, non-zero means turns until active
        self.refractory_shoot = 0.0
        self.refractory_div = expected_donor_A_div_refractory_turns  # floor(0.50 / time_per_turn)  # OLD VERSION: expected_donor_div_time/4/time_per_turn # refractory period after division in turns
        self.div_time_fraction = 1.0  # 0.5  # 1.0 is default, 0.0 is 100% success every turn
        self.shoot_time_fraction = 1.0  # 0.5  # 1.0 is default, 0.0 is 100% success every turn
        self.div_time = expected_donor_A_div_time
        self.shoot_time = expected_donor_A_shoot_time
        self.targets_min = 0
        self.targets_max = 5


class DonorTypeB(Cell):
    def __init__(self, location):
        Cell.__init__(self, 'D_b', location)
        self.pause = 0  # 0 if cell is active, non-zero means turns until active
        self.refractory_shoot = 0.0
        self.refractory_div = expected_donor_B_div_refractory_turns  # OLD VERSION: expected_donor_div_time/4/time_per_turn # refractory period after division in turns
        self.div_time_fraction = 1.0  # 1.0 is default, 0.0 is 100% success every turn
        self.shoot_time_fraction = 1.0  # 1.0 is default, 0.0 is 100% success every turn
        self.div_time = expected_donor_B_div_time
        self.shoot_time = expected_donor_B_shoot_time
        self.targets_min = 0
        self.targets_max = 5


class Debris(Cell):
    def __init__(self, location):
        Cell.__init__(self, 'B', location)
        self.decay_timer = debris_decay_time  # time (in hours) until debris decays


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


def build_lattice_testing():
    pivot = n/5
    anti_pivot = n - pivot - 1
    lattice[pivot][pivot] = DonorTypeA([pivot, pivot])
    lattice[anti_pivot][anti_pivot] = DonorTypeB([anti_pivot, anti_pivot])
    print "WARNING - testing lattice in use"
    return lattice


def build_lattice_random(seed=5):
    # seed: determines ratio of donors to recipients for random homogeneous conditions
    random_lattice = randint(seed, size=(n, n))
    for i in xrange(n):
        for j in xrange(n):
            m = random_lattice[i][j]
            if m == 0:
                lattice[i][j] = DonorTypeA([i, j])
            elif m == 1:
                lattice[i][j] = DonorTypeB([i, j])
            elif m in range(2, seed):
                lattice[i][j] = Empty([i, j])
    print random_lattice, "\n"
    return


def build_lattice_diag():
    for i in xrange(n):
        for j in xrange(n):
            if j < i:
                lattice[i][j] = DonorTypeA([i, j])
            else:
                lattice[i][j] = DonorTypeB([i, j])
    return


def build_lattice_concentric():
    radius_inner = np.ceil(n * 0.25)
    for i in xrange(n):
        for j in xrange(n):
            if np.sqrt((i - n/2)**2 + (j - n/2)**2) <= radius_inner:
                lattice[i][j] = DonorTypeA([i, j])
            else:
                lattice[i][j] = DonorTypeB([i, j])
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


def divide(cell, empty_neighbours, new_cell_locations, dict_counts):
    distr = randint(0, 100)
    success = 0  # note that successful division = 1
    if distr * cell.div_time_fraction < 100.0 / turn_rate and len(empty_neighbours) > 0:
        success = 1
        daughter_loc = random.choice(empty_neighbours)
        if 'D_a' == cell.label:
            lattice[daughter_loc[0]][daughter_loc[1]] = DonorTypeA(daughter_loc)
            #cell.maturity = floor(cell.maturity / 2)
        elif 'D_b' == cell.label:
            lattice[daughter_loc[0]][daughter_loc[1]] = DonorTypeB(daughter_loc)
        else:
            raise Exception("Illegal cell type")
        cell.pause = cell.refractory_div
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
            lattice[target_loc[0]][target_loc[1]] = Debris(target_loc)
            #new_cell_locations.append(target_loc)
            dict_counts['B'] += 1
            dict_counts[target.label] -= 1
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
                cell.decay_timer -= time_per_turn
                if cell.decay_timer <= 0:
                    lattice[loc[0]][loc[1]] = Empty(loc)
                    locations_to_remove.append(loc)
                    dict_counts['B'] -= 1
                    dict_counts['_'] += 1

            # donor behaviour
            elif is_donor_type_a(loc) or is_donor_type_b(loc):
                if 0.0 < cell.pause:  # if paused, decrement pause timer
                    cell.pause -= 1
                else:
                    empty_neighbours = cell.get_label_surroundings('_', search_radius_bacteria)
                    # TRY TO DIVIDE BEFORE SHOOTING
                    if not divide(cell, empty_neighbours, new_cell_locations, dict_counts):
                        shoot(cell, dict_counts)

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
    build_lattice_random()
    #build_lattice_testing()
    #build_lattice_diag()
    #build_lattice_concentric()

    run_sim()

    # write data to file
    data_name = "lattice_data.csv"
    data_file = data_folder + data_name
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

    print "\nDone!"
    return

if __name__ == '__main__':
    main()
