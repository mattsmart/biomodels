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

SPEED
-instead of explicit class structure for the states, could just use DICT (may be faster)
-use location tuples instead of lists (faster assigning)
-faster and better probability modules
-all to numpy arrays

-finally utilize inheritance, have the 3 subclasses
-should we incorporate cell death

BUGS
-potential bug if empty cells are not carefully initiated (can't distinguish between unused spot and 'Empty' cell)
    -might be able to turn this into speedup by having not used == empty
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
n = 100
search_radius_square = 2
assert search_radius_square < n / 2
seed = 5
standard_run_time = 24.0  # typical simulation time in h

# division and recovery times
div_time_staph = 0.5
div_time_ecoli = 0.5
expected_receiver_div_time = div_time_ecoli  # avg time for 1 R cell to divide in h
expected_donor_div_time = div_time_ecoli  # avg time for 1 D cell to divide in h

# conjugation rate
conj_super_rate = 1.0
conj_ecoli_rate = 10.0
conj_staph_rate = 10 ** 4.0
expected_conj_time = conj_ecoli_rate  # avg time for 1 cell to conjugate in h (Jama: 10h for e.coli, more for staph)

death_rate_lab = 0.0  # for now
death_rate_body = 4.0  # average h for 1 cell to die to immune (?) cells (NOT IMPLEMENTED)

turn_rate = 2.0  # average turns between each division; simulation step size
time_per_turn = expected_receiver_div_time / turn_rate

times_antibiotic = []  # antibiotic introduction time in h
turns_antibiotic = [x / time_per_turn for x in times_antibiotic]


# Classes
# =================================================
# represents the state of a lattice cell: empty, donor, recipient
class Cell(object):
    def __init__(self, label, location):
        self.label = label  # symbol; either "(_) Empty, (R)eceiver, (D)onor"
        self.location = location  # list [i,j]

    def __str__(self):
        return self.label

    # given cell # i,j, returns a list of array pairs (up to 6) rep nearby lattice cells
    def get_surroundings(self):
        location = self.location
        row = location[0]
        col = location[1]
        is_odd = row % 2  # 0 means even row, 1 means odd row  # TODO REMOVE make for PBC though

        up = [row - 1, col]  # note the negative
        down = [row + 1, col]
        left = [row, col - 1]
        right = [row, col + 1]
        up_left = [row - 1, col - 1]
        up_right = [row - 1, col + 1]
        down_left = [row + 1, col - 1]
        down_right = [row + 1, col + 1]

        if row == 0:  # Top Case (even row)
            if col == 0:
                surroundings = [right, down]
            elif col == (n - 1):
                surroundings = [left, down, down_left]
            else:
                surroundings = [left, right, down_left, down]

        elif row == (n - 1):  # Bottom Case
            if is_odd:
                if col == 0:
                    surroundings = [up, right, up_right]
                elif col == (n - 1):
                    surroundings = [left, up]
                else:
                    surroundings = [left, right, up, up_right]
            else:
                if col == 0:
                    surroundings = [right, up]
                elif col == (n - 1):
                    surroundings = [left, up, up_left]
                else:
                    surroundings = [left, right, up, up_left]

        elif col == 0:  # Left Case
            if is_odd:
                surroundings = [right, up, down, up_right, down_right]
            else:
                surroundings = [right, up, down]

        elif col == (n - 1):  # Right Case
            if is_odd:
                surroundings = [left, up, down]
            else:
                surroundings = [left, up, down, up_left, down_left]

        else:  # General Case
            if is_odd:
                surroundings = [left, right, up, down, up_right, down_right]
            else:
                surroundings = [left, right, up, down, up_left, down_left]

        return surroundings

    # NEW NEIGHBOURS FUNCTION
    def get_surroundings_square(self, search_radius):
        """Specifies the location of the top left corner of the search square
        Args:
            search_radius: half-edge length of the square
        Returns:
            list of locations; length should be (2 * search_radius + 1) ** 2
        Notes:
            - periodic BCs apply, so search boxes wrap around at boundaries
            - note that we assert that search_radius be less than half the grid size
        """
        # utility variables
        row = self.location[0]
        col = self.location[1]

        # intiialize list for speedup
        # TODO surroundings = [0] * (search_radius ** 2)
        surroundings = [(row_to_search % n, col_to_search % n)
                        for row_to_search in xrange(row - search_radius, row + search_radius + 1)
                        for col_to_search in xrange(col - search_radius, col + search_radius + 1)]
        # assert len(surroundings) == (2 * search_radius + 1) ** 2
        return surroundings

    def get_label_surroundings(self, cell_label):
        if cell_label not in ['_', 'R', 'D']:
            raise Exception("Illegal cell label (_, R, or D)")
        neighbours = self.get_surroundings_square(search_radius=search_radius_square)
        neighbours_of_specified_type = []
        for loc in neighbours:
            if cell_label == lattice[loc[0]][loc[1]].label:
                neighbours_of_specified_type.append(loc)
        return neighbours_of_specified_type


class Empty(Cell):
    def __init__(self, location):
        Cell.__init__(self, '_', location)


class Receiver(Cell):
    def __init__(self, location):
        Cell.__init__(self, 'R', location)
        self.pause = 0  # 0 if cell is active, non-zero means turns until active
        self.refractory_div = expected_donor_div_time / 4 / time_per_turn  # refractory period after division in turns
        self.resistance_factor = 1  # resistance to antibiotics


class Donor(Cell):
    def __init__(self, location):
        Cell.__init__(self, 'D', location)
        self.pause = 0  # 0 if cell is active, non-zero means turns until active
        self.maturity = 0  # starting probability of conjugation
        self.maxmaturity = 50  # max probability of conjugation
        self.refractory_conj = ceil(
            0.25 / time_per_turn)  # OLD VERSION: expected_conj_time/16/time_per_turn # refractory period after conjugation in turns
        self.refractory_div = ceil(
            0.50 / time_per_turn)  # OLD VERSION: expected_donor_div_time/4/time_per_turn # refractory period after division in turns
        self.resistance_factor = 1;  # resistance to antibiotics


# Initiate Cell Lattice and Data Directory
# =================================================
lattice = [[Empty([x, y]) for y in xrange(n)] for x in xrange(n)]
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
    lattice[pivot][pivot] = Receiver([pivot, pivot])
    lattice[anti_pivot][anti_pivot] = Donor([anti_pivot, anti_pivot])
    print lattice
    print "WARNING - testing lattice in use"
    return lattice


def build_lattice_random():
    random_lattice = randint(seed, size=(n, n))
    for i in xrange(n):
        for j in xrange(n):
            m = random_lattice[i][j]
            if m == 0:
                lattice[i][j] = Receiver([i, j])
            elif m == 1:
                lattice[i][j] = Donor([i, j])
            elif m in range(2, seed):
                lattice[i][j] = Empty([i, j])
    print random_lattice, "\n"
    return


def is_empty(loc):
    return '_' == lattice[loc[0]][loc[1]].label


def is_receiver(loc):
    return 'R' == lattice[loc[0]][loc[1]].label


def is_donor(loc):
    return 'D' == lattice[loc[0]][loc[1]].label


def divide(cell, empty_neighbours):
    distr = randint(0, 100)
    success = 0  # successful division = 1
    if distr < 100.0 / turn_rate:
        success = 1
        daughter_loc = random.choice(empty_neighbours)
        if 'D' == cell.label:
            lattice[daughter_loc[0]][daughter_loc[1]] = Donor(daughter_loc)
            cell.maturity = floor(cell.maturity / 2)
        elif 'R' == cell.label:
            lattice[daughter_loc[0]][daughter_loc[1]] = Receiver(daughter_loc)
        else:
            raise Exception("Illegal cell type")
        cell.pause = cell.refractory_div
    return success


def conjugate(cell, receiver_neighbours):
    distr = randint(0, 1000)  # [1, 1000]
    success = 0  # successful conjugation = 1
    conj_rate_rel_div_rate = expected_conj_time / expected_donor_div_time
    if distr < (1000.0 / turn_rate) / conj_rate_rel_div_rate:
        success = 1
        daughter_loc = random.choice(receiver_neighbours)
        lattice[daughter_loc[0]][daughter_loc[1]] = Donor(daughter_loc)
        cell.pause = cell.refractory_conj
    return success


def count_cells():  # returns a list of current cell counts: [# of empty, # of receiver, # of donor]
    E = 0
    R = 0
    D = 0
    for i in xrange(n):
        for j in xrange(n):
            loc = [i, j]
            if is_receiver(loc):
                R += 1
            elif is_donor(loc):
                D += 1
            else:
                E += 1
    return [E, R, D]


def run_sim(T):  # T = total sim time

    # get stats for lattice initial condition before entering simulation loop
    [E, R, D] = count_cells()
    lattice_data.append([0, 0.0, E, R, D])
    lattice_plotter(lattice, 0.0, n, plot_lattice_folder)

    # begin simulation
    turns = int(ceil(T / time_per_turn))
    for t in xrange(turns):
        print 'Turn ', t, ' : Time Elapsed ', t * time_per_turn, "h"
        # printer()
        for i in xrange(n):
            for j in xrange(n):
                cell = lattice[i][j]
                loc = [i, j]

                if is_empty(loc):
                    continue

                # INSERT prob of cell death

                if is_receiver(loc):
                    if 0 < cell.pause:  # if paused, reduce pause time
                        cell.pause -= 1
                        continue

                    receiver_neighbours = cell.get_label_surroundings('R')
                    empty_neighbours = cell.get_label_surroundings('_')
                    if not empty_neighbours:  # skip if surrounded
                        continue
                    else:  # chance to divide
                        divide(cell, empty_neighbours)
                        continue

                elif is_donor(loc):
                    if cell.maturity < cell.maxmaturity:  #####COME BACK TO THIS
                        cell.maturity += 10
                    if 0 < cell.pause:  # if paused, reduce pause time
                        cell.pause -= 1
                        continue

                    receiver_neighbours = cell.get_label_surroundings('R')
                    empty_neighbours = cell.get_label_surroundings('_')
                    if not empty_neighbours:  # surrounded?
                        if not receiver_neighbours:  # no R neighbours?
                            continue
                        else:  # R neighbours
                            conjugate(cell, receiver_neighbours)
                            continue
                    else:  # some empty neighbours
                        if not receiver_neighbours:  # no R neighbours
                            divide(cell, empty_neighbours)
                            continue
                        else:  # some R neighbours
                            if 1 == randint(1, 3):  # divide first
                                if not divide(cell, empty_neighbours):
                                    conjugate(cell, receiver_neighbours)
                                continue
                            else:  # conjugate first
                                if not conjugate(cell, receiver_neighbours):
                                    divide(cell, empty_neighbours)
                                continue

                else:
                    raise Exception("Not _, R, or D")
                    break

        # get lattice stats for this timestep
        [E, R, D] = count_cells()
        lattice_data.append([(t + 1), (t + 1) * time_per_turn, E, R, D])
        # print lattice_data, "\n"
        lattice_plotter(lattice, (t + 1) * time_per_turn, n, plot_lattice_folder)
        # raw_input()

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
    data_dict = {'iters': iters,
                 'time': time,
                 'E': E,
                 'R': R,
                 'D': D}

    data_plotter(data_dict, data_file, plot_data_folder)

    print "\n oops"
    return

if __name__ == '__main__':
    main()
