import csv
import datetime
import os
import random
from math import ceil, floor
from numpy.random import randint

from plot_data import data_plotter
from plot_lattice import lattice_plotter


"""
TODO
-finally utilize inheritance, have the 3 subclasses
-should we incorporate cell death
-make conj rate nutrient dependent
-remove or augment the maturity module
-ICs that resemble the PDEs

SPEED
-instead of explicit class structure for the states, could just use DICT (may be faster)
-use location tuples instead of lists (faster assigning)
-faster and better probability modules
-all to numpy arrays
-iterate only over nonempty cells

% PLOTTING SPEED
-print plot every kth turn
-make plotting faster or save big matrices for plotting later?
-dont plot text in every cell for 10k cells LOL
-might be a faster plotting package

BUGS
-potential bug if empty cells are not carefully initiated (can't distinguish between unused spot and 'Empty' cell)
    -might be able to turn this into speedup by having not used == empty
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
n = 100

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
plots_period_in_turns = 2 * turn_rate


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
        # intiialize list for speedup
        # TODO surroundings = [0] * (search_radius ** 2)
        surroundings = [(row_to_search % n, col_to_search % n)
                        for row_to_search in xrange(row - search_radius, row + search_radius + 1)
                        for col_to_search in xrange(col - search_radius, col + search_radius + 1)]
        # assert len(surroundings) == (2 * search_radius + 1) ** 2
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
        location_layers_nutrients = [(0, 0)] * (max_nutrient_radius + 1)
        for radius in xrange(max_nutrient_radius + 1):
            surroundings = self.get_surroundings_square(radius)
            location_layers_nutrients[radius] = surroundings

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
            for row, col in nutrient_location_layer:
                if lattice[row][col].nutrients > 0:
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
                        nutrient_location_layer.remove(loc)
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
        self.maturity = 0  # starting probability of conjugation
        self.maxmaturity = 50  # max probability of conjugation
        self.refractory_conj = ceil(0.25 / time_per_turn)  # OLD VERSION: expected_conj_time/16/time_per_turn # refractory period after conjugation in turns
        self.refractory_div = ceil(0.50 / time_per_turn)  # OLD VERSION: expected_donor_div_time/4/time_per_turn # refractory period after division in turns


# Initiate Cell Lattice and Data Directory
# =================================================
lattice = [[Empty([x, y], nutrient_initial_condition) for y in xrange(n)] for x in xrange(n)]
lattice_data = []  # list of list, sublists are [iters, time, E, R, D]


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


def get_nutrients(loc):
    return lattice[loc[0]][loc[1]].nutrients


def divide(cell, empty_neighbours, new_cell_locations):
    nutrients_are_available = cell.is_nutrient_available()
    distr = randint(0, 100)
    success = 0  # successful division = 1
    if distr < 100.0 / turn_rate and nutrients_are_available and len(empty_neighbours) > 0:
        success = 1
        daughter_loc = random.choice(empty_neighbours)
        cell.choose_and_exhaust_nutrient()
        daughter_loc_nutrients = get_nutrients(daughter_loc)
        if 'D' == cell.label:
            lattice[daughter_loc[0]][daughter_loc[1]] = Donor(daughter_loc, daughter_loc_nutrients)
            cell.maturity = floor(cell.maturity / 2)
        elif 'R' == cell.label:
            lattice[daughter_loc[0]][daughter_loc[1]] = Receiver(daughter_loc, daughter_loc_nutrients)
        else:
            raise Exception("Illegal cell type")
        cell.pause = cell.refractory_div
        new_cell_locations.append(daughter_loc)
    return success


def conjugate(cell, recipient_neighbours):
    distr = randint(0, 1000)  # [1, 1000]
    success = 0  # successful conjugation = 1
    conj_rate_rel_div_rate = expected_conj_time / expected_donor_div_time
    if distr < (1000.0 / turn_rate) / conj_rate_rel_div_rate and len(recipient_neighbours) > 0:
        success = 1
        mate_loc = random.choice(recipient_neighbours)
        lattice[mate_loc[0]][mate_loc[1]] = Donor(mate_loc, get_nutrients(mate_loc))
        cell.pause = cell.refractory_conj
    return success


def count_cells():  # returns a list of current cell counts: [# of empty, # of recipient, # of donor, # of nutrients]
    E = 0
    R = 0
    D = 0
    N = 0
    for i in xrange(n):
        for j in xrange(n):
            loc = [i, j]
            N += get_nutrients(loc)
            if is_recipient(loc):
                R += 1
            elif is_donor(loc):
                D += 1
            else:
                E += 1
    return [E, R, D, N]


def get_cell_locations():
    cell_locations = []
    for i in xrange(n):
        for j in xrange(n):
            loc = [i, j]
            if is_donor(loc) or is_recipient(loc):
                cell_locations.append(loc)
    return cell_locations


def run_sim(T):  # T = total sim time

    # get stats for lattice initial condition before entering simulation loop
    [E, R, D, N] = count_cells()
    lattice_data.append([0, 0.0, E, R, D, N])
    lattice_plotter(lattice, 0.0, n, plot_lattice_folder)

    # begin simulation
    new_cell_locations = []
    cell_locations = get_cell_locations()
    turns = int(ceil(T / time_per_turn))
    for turn in xrange(1, turns + 1):
        print 'Turn ', turn, ' : Time Elapsed ', turn * time_per_turn, "h"
        # printer()
        cell_locations = cell_locations + new_cell_locations
        new_cell_locations = []
        for loc in cell_locations:
            cell = lattice[loc[0]][loc[1]]
            if 0 < cell.pause:  # if paused, decrement pause timer
                cell.pause -= 1
            else:
                empty_neighbours = cell.get_label_surroundings('_', search_radius_bacteria)

                # recipient behaviour
                if is_recipient(loc):
                    divide(cell, empty_neighbours, new_cell_locations)

                # donor behaviour
                elif is_donor(loc):
                    if cell.maturity < cell.maxmaturity:
                        cell.maturity += 10
                    recipient_neighbours = cell.get_label_surroundings('R', search_radius_bacteria)
                    no_division_flag = not empty_neighbours
                    no_conjugation_flag = not recipient_neighbours
                    if no_division_flag and no_conjugation_flag:
                        pass
                    elif no_division_flag:
                        conjugate(cell, recipient_neighbours)
                    elif no_conjugation_flag:
                        divide(cell, empty_neighbours, new_cell_locations)
                    else:  # chance to either conjugate or divide, randomize the order of potential events
                        if 1 == randint(1, 3):  # try to divide first (33% of the time)
                            if not divide(cell, empty_neighbours, new_cell_locations):
                                conjugate(cell, recipient_neighbours)
                        else:  # try to conjugate first
                            if not conjugate(cell, recipient_neighbours):
                                divide(cell, empty_neighbours, new_cell_locations)
                else:
                    print "WARNING - Cell not R or D"
                    raise Exception("Cell not R or D")

        # get lattice stats for this timestep
        [E, R, D, N] = count_cells()
        lattice_data.append([turn, turn * time_per_turn, E, R, D, N])
        # periodically plot the lattice (it takes a while)
        if turn % plots_period_in_turns == 0:
            lattice_plotter(lattice, turn * time_per_turn, n, plot_lattice_folder)

    return lattice_data


# Main Function
# =================================================
def main():
    build_lattice_random()
    #build_lattice_testing()
    run_sim(standard_run_time)

    data_name = "lattice_data.csv"
    data_file = data_folder + data_name
    with open(data_file, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(lattice_data)

    # convert lattice_data to a dictionary
    iters = [x[0] for x in lattice_data]
    time = [x[1] for x in lattice_data]
    E = [x[2] for x in lattice_data]
    R = [x[3] for x in lattice_data]
    D = [x[4] for x in lattice_data]
    N = [x[5] for x in lattice_data]
    data_dict = {'iters': iters,
                 'time': time,
                 'E': E,
                 'R': R,
                 'D': D,
                 'N': N}

    data_plotter(data_dict, data_file, plot_data_folder)

    print "\nDone!"
    return

if __name__ == '__main__':
    main()
