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
-make conj rate nutrient dependent
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

BUGS
-should skip self when going through surroundings, but not nutrients (i.e. dont count your current position)
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
seed = 5  # determines ratio of donors to recipients for random homogeneous conditions
search_radius_bacteria = 1
max_search_radius_nutrient = 3
assert search_radius_bacteria < n / 2 and max_search_radius_nutrient < n / 2

# nutrient settings
nutrient_initial_condition = 1 # 10

# division and recovery times
div_time_staph = 0.5
div_time_ecoli = 0.5
expected_recipient_div_time = div_time_ecoli  # avg time for 1 R cell to divide in h
expected_donor_div_time = div_time_ecoli  # avg time for 1 D cell to divide in h

# conjugation rate
conj_super_rate = 1.0
conj_ecoli_rate = 10.0
conj_staph_rate = 10 ** 4.0
expected_conj_time = conj_ecoli_rate  # avg time for 1 cell to conjugate in h (Jama: 10h for e.coli, more for staph)

# general cell settings
death_rate_lab = 0.0

# simulation time settings
standard_run_time = 24.0  # typical simulation time in h
turn_rate = 2.0  # average turns between each division; simulation step size
time_per_turn = expected_recipient_div_time / turn_rate
plots_period_in_turns = 2 * turn_rate  # 1 or 1000 or 2 * turn_rate
total_turns = int(ceil(standard_run_time / time_per_turn))


# Classes
# =================================================
# represents the state of a lattice cell: empty, donor, recipient
class Cell(object):
    def __init__(self, label, location, nutrients):
        self.label = label  # symbol; either "(_) Empty, (R)eceiver, (D)onor"
        self.location = location  # list [i,j]
        self.nutrients = nutrients

    def __str__(self):
        return self.label

    def deplete_nutrients(self):
        assert self.nutrients > 0
        self.nutrients -= 1

    def get_surroundings_square(self, search_radius):
        """Specifies the location of the top left corner of the search square
        Args:
            search_radius: half-edge length of the square
        Returns:
            list of locations; length should be (2 * search_radius + 1) ** 2 (- 1 remove self?)
        Notes:
            - periodic BCs apply, so search boxes wrap around at boundaries
            - note that we assert that search_radius be less than half the grid size
            - may have different search radius depending om context (neighbouring bacteria / empty cells / nutrient)
            - currently DOES NOT remove the original location
        """
        row = self.location[0]
        col = self.location[1]
        surroundings = [(row_to_search % n, col_to_search % n)
                        for row_to_search in xrange(row - search_radius, row + search_radius + 1)
                        for col_to_search in xrange(col - search_radius, col + search_radius + 1)]
        return surroundings

    def get_label_surroundings(self, cell_label, search_radius):
        if cell_label not in ['_', 'R', 'D']:
            raise Exception("Illegal cell label (_, R, or D)")
        neighbours = self.get_surroundings_square(search_radius=search_radius)
        neighbours_of_specified_type = []
        for loc in neighbours:  # TODO should skip self when going through (i.e. don't count your current position)
            if cell_label == lattice[loc[0]][loc[1]].label:
                neighbours_of_specified_type.append(loc)
        return neighbours_of_specified_type

    def get_nutrient_surroundings_ordered(self, max_nutrient_radius):
        """Gives a list of POSSIBLY remaining nutrient surroundings (square), ordered by increasing radii
        Args:
            max_nutrient_radius: int (e.g. 2)
        Returns:
            ordered list of groups of locations of nutrients which may or may not be available,
            specifically: [[radius=0 nutrient locs], [radius=1 nutrient locs], ...,  [radius=max nutrient locs]]
        Notes:
            - very VERY inefficient algorithm, should clean it up
        """
        # create initial surroundings, with duplicates
        location_layers_nutrients = [[0]] * (max_nutrient_radius + 1)
        for radius in xrange(max_nutrient_radius + 1):
            location_layers_nutrients[radius] = self.get_surroundings_square(radius)  # TODO speed this up later

        # remove duplicates
        for radius in xrange(max_nutrient_radius, 1, -1):
            for loc in location_layers_nutrients[radius-1]:
                location_layers_nutrients[radius].remove(loc)

        return location_layers_nutrients

    def is_nutrient_available(self):
        """Checks if any nutrients are available within the search radius
        """
        nutrient_layers = self.get_nutrient_surroundings_ordered(max_search_radius_nutrient)
        for nutrient_location_layer in nutrient_layers:
            for loc in nutrient_location_layer:
                if lattice[loc[0]][loc[1]].nutrients > 0:
                    return True
        return False

    def choose_and_exhaust_nutrient(self):
        """Chooses a random nutrient location to deplete by a value of 1
        Notes:
            - starts with locations closest to the cell (radius=0) and moves outwards
        """
        nutrient_layers = self.get_nutrient_surroundings_ordered(max_search_radius_nutrient)
        while len(nutrient_layers) > 0:
            for nutrient_location_layer in nutrient_layers:
                if len(nutrient_location_layer) > 0:
                    loc = random.choice(nutrient_location_layer)
                    if lattice[loc[0]][loc[1]].nutrients > 0:
                        lattice[loc[0]][loc[1]].deplete_nutrients()
                        return
                    else:
                        nutrient_location_layer.remove(loc)  # TODO possible bug
                else:
                    nutrient_layers.remove(nutrient_location_layer)
        print "WARNING - choosing to exhaust nutrients when none are available, continuing anyways"
        return


class Empty(Cell):
    def __init__(self, location, nutrients):
        Cell.__init__(self, '_', location, nutrients)


class Receiver(Cell):
    def __init__(self, location, nutrients):
        Cell.__init__(self, 'R', location, nutrients)
        self.pause = 0  # 0 if cell is active, non-zero means turns until active
        self.refractory_div = expected_donor_div_time / 4 / time_per_turn  # refractory period after division in turns


class Donor(Cell):
    def __init__(self, location, nutrients):
        Cell.__init__(self, 'D', location, nutrients)
        self.pause = 0  # 0 if cell is active, non-zero means turns until active
        #self.maturity = 0  # starting probability of conjugation
        #self.maxmaturity = 50  # max probability of conjugation
        self.refractory_conj = ceil(0.25 / time_per_turn)  # OLD VERSION: expected_conj_time/16/time_per_turn # refractory period after conjugation in turns
        self.refractory_div = ceil(0.50 / time_per_turn)  # OLD VERSION: expected_donor_div_time/4/time_per_turn # refractory period after division in turns


class Transconjugant(Cell):
    def __init__(self, location, nutrients):
        Cell.__init__(self, 'T', location, nutrients)
        self.pause = 0  # 0 if cell is active, non-zero means turns until active
        #self.maturity = 0  # starting probability of conjugation
        #self.maxmaturity = 50  # max probability of conjugation
        self.refractory_conj = ceil(0.25 / time_per_turn)  # OLD VERSION: expected_conj_time/16/time_per_turn # refractory period after conjugation in turns
        self.refractory_div = ceil(0.50 / time_per_turn)  # OLD VERSION: expected_donor_div_time/4/time_per_turn # refractory period after division in turns


# Initiate Cell Lattice and Data Directory
# =================================================
lattice = [[Empty([x, y], nutrient_initial_condition) for y in xrange(n)] for x in xrange(n)]  # this can be made faster as np array
lattice_data = np.zeros((total_turns + 1, 7))  # sublists are [turn, time, E, R, D, T, N]


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
    lattice[pivot][pivot] = Receiver([pivot, pivot], lattice[pivot][pivot].nutrients)
    lattice[anti_pivot][anti_pivot] = Donor([anti_pivot, anti_pivot], lattice[anti_pivot][anti_pivot].nutrients)
    print "WARNING - testing lattice in use"
    return lattice


def build_lattice_random():
    random_lattice = randint(seed, size=(n, n))
    for i in xrange(n):
        for j in xrange(n):
            m = random_lattice[i][j]
            if m == 0:
                lattice[i][j] = Receiver([i, j], nutrient_initial_condition)
            elif m == 1:
                lattice[i][j] = Donor([i, j], nutrient_initial_condition)
            elif m in range(2, seed):
                lattice[i][j] = Empty([i, j], nutrient_initial_condition)
    print random_lattice, "\n"
    return


def is_empty(loc):
    return '_' == lattice[loc[0]][loc[1]].label


def is_recipient(loc):
    return 'R' == lattice[loc[0]][loc[1]].label


def is_donor(loc):
    return 'D' == lattice[loc[0]][loc[1]].label


def is_transconjugant(loc):
    return 'T' == lattice[loc[0]][loc[1]].label


def get_nutrients(loc):
    return lattice[loc[0]][loc[1]].nutrients


def get_label(loc):
    return lattice[loc[0]][loc[1]].label


def divide(cell, empty_neighbours, new_cell_locations, dict_counts):
    nutrients_are_available = cell.is_nutrient_available()
    distr = randint(0, 100)
    success = 0  # note that successful division = 1
    if distr < 100.0 / turn_rate and nutrients_are_available and len(empty_neighbours) > 0:
        success = 1
        daughter_loc = random.choice(empty_neighbours)
        cell.choose_and_exhaust_nutrient()
        daughter_loc_nutrients = get_nutrients(daughter_loc)
        if 'D' == cell.label:
            lattice[daughter_loc[0]][daughter_loc[1]] = Donor(daughter_loc, daughter_loc_nutrients)
            #cell.maturity = floor(cell.maturity / 2)
        elif 'T' == cell.label:
            lattice[daughter_loc[0]][daughter_loc[1]] = Transconjugant(daughter_loc, daughter_loc_nutrients)
            #cell.maturity = floor(cell.maturity / 2)
        elif 'R' == cell.label:
            lattice[daughter_loc[0]][daughter_loc[1]] = Receiver(daughter_loc, daughter_loc_nutrients)
        else:
            raise Exception("Illegal cell type")
        cell.pause = cell.refractory_div
        # update tracking variables
        new_cell_locations.append(daughter_loc)
        dict_counts[cell.label] += 1
        dict_counts['N'] -= 1
        dict_counts['_'] -= 1
    return success


def conjugate(cell, recipient_neighbours, dict_counts):
    distr = randint(0, 1000)  # [1, 1000]
    success = 0  # note that successful conjugation = 1
    conj_rate_rel_div_rate = expected_conj_time / expected_donor_div_time
    if distr < (1000.0 / turn_rate) / conj_rate_rel_div_rate and len(recipient_neighbours) > 0:
        success = 1
        mate_loc = random.choice(recipient_neighbours)
        lattice[mate_loc[0]][mate_loc[1]] = Transconjugant(mate_loc, get_nutrients(mate_loc))
        cell.pause = cell.refractory_conj
        # update tracking variables
        dict_counts['T'] += 1
        dict_counts['R'] -= 1
    return success


def count_cells():  # returns a dict of current cell counts: [# of empty, # of recipient, # of donor, # of nutrients]
    keys = ['_', 'R', 'D', 'T', 'N']
    counts = {key: 0 for key in keys}
    for i in xrange(n):
        for j in xrange(n):
            loc = (i, j)
            counts['N'] += get_nutrients(loc)
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
    lattice_data[0, :] = np.array([0, 0.0, dict_counts['_'], dict_counts['R'], dict_counts['D'], dict_counts['T'], dict_counts['N']])

    # plot initial conditions
    #lattice_plotter(lattice, 0.0, n, dict_counts, plot_lattice_folder)  # TODO Re-add

    # simulation loop initialization
    new_cell_locations = []
    cell_locations = get_cell_locations()

    for turn in xrange(1, total_turns + 1):
        print '\nTurn ', turn, ' : Time Elapsed ', turn * time_per_turn, "h"
        cell_locations = cell_locations + new_cell_locations
        new_cell_locations = []

        # timestep profiling
        t0_a = time.clock()
        t0_b = time.time()

        for loc in cell_locations:
            cell = lattice[loc[0]][loc[1]]
            if 0 < cell.pause:  # if paused, decrement pause timer
                cell.pause -= 1
            else:
                empty_neighbours = cell.get_label_surroundings('_', search_radius_bacteria)

                # recipient behaviour
                if is_recipient(loc):
                    divide(cell, empty_neighbours, new_cell_locations, dict_counts)

                # donor behaviour
                elif is_donor(loc) or is_transconjugant(loc):
                    #  if cell.maturity < cell.maxmaturity:
                    #      cell.maturity += 10
                    recipient_neighbours = cell.get_label_surroundings('R', search_radius_bacteria)
                    no_division_flag = not empty_neighbours
                    no_conjugation_flag = not recipient_neighbours
                    if no_division_flag and no_conjugation_flag:
                        pass
                    elif no_division_flag:
                        conjugate(cell, recipient_neighbours, dict_counts)
                    elif no_conjugation_flag:
                        divide(cell, empty_neighbours, new_cell_locations, dict_counts)
                    else:  # chance to either conjugate or divide, randomize the order of potential events
                        if 1 == randint(1, 3):  # try to divide first (33% of the time)
                            if not divide(cell, empty_neighbours, new_cell_locations, dict_counts):
                                conjugate(cell, recipient_neighbours, dict_counts)
                        else:  # try to conjugate first
                            if not conjugate(cell, recipient_neighbours, dict_counts):
                                divide(cell, empty_neighbours, new_cell_locations, dict_counts)

                else:
                    print "WARNING - Cell not R or D or T"
                    raise Exception("Cell not R or D or T")

        # get lattice stats for this timestep
        counts = count_cells()
        lattice_data[turn, :] = np.array([turn, turn * time_per_turn, dict_counts['_'], dict_counts['R'], dict_counts['D'], dict_counts['T'], dict_counts['N']])
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
    build_lattice_random()
    #build_lattice_testing()
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
                 'R': lattice_data[:, 3],
                 'D': lattice_data[:, 4],
                 'T': lattice_data[:, 5],
                 'N': lattice_data[:, 6]}
    data_plotter(data_dict, data_file, plot_data_folder)

    print "\nDone!"
    return

if __name__ == '__main__':
    main()
