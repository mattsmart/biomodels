import csv
import datetime
import os
import random
from math import ceil, floor
from numpy.random import randint

#from plot_data import data_plotter  # TODO change names to prevent misinterpretation with simple abm
#from plot_lattice import lattice_plotter  # TODO change names to prevent misinterpretation with simple abm


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
n = 10 # TODO FIX 100
seed = 5
standard_run_time = 24.0  # typical simulation time in h

# cody: 0.5 h for staph, 0.25 h for e.coli (back to 0.5h ecoli -oct 17)
div_time_staph = 0.5
div_time_ecoli = 0.5
expected_receiver_div_time = div_time_ecoli  # avg time for 1 R cell to divide in h
expected_donor_div_time = div_time_ecoli  # avg time for 1 D cell to divide in h

# cody: staph rate could be 10^4 i.e. 10^-4
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
# represents the state of a grid cell: empty, donor, recipient
class Cell(object):
    def __init__(self, label, location):
        self.label = label  # symbol; either "(_) Empty, (R)eceiver, (D)onor"
        self.location = location  # list [i,j]

    def __str__(self):
        return self.label

    # given cell # i,j, returns a list of array pairs (up to 6) rep nearby lattice cells
    # COMMENTS: Do I need all the return statements? or just one at the end?
    def get_surroundings(self):
        location = self.location
        row = location[0]
        col = location[1]
        is_odd = row % 2  # 0 means even row, 1 means odd row

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

    def get_label_surroundings(self, cell_label):
        if cell_label not in ['_', 'R', 'D']:
            raise Exception("Illegal cell label (_, R, or D)")
        neighbours = self.get_surroundings()
        empty_neighbours = []
        for loc in neighbours:
            if cell_label == grid[loc[0]][loc[1]].label:
                empty_neighbours.append(loc)
        return empty_neighbours


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


# Initiate Cell Grid and Data Directory
# =================================================
grid = [[Empty([x, y]) for y in xrange(n)] for x in xrange(n)]
grid_data = []  # list of list, sublists are [iters, time, E, R, D]


# Functions
# =================================================

def printer():
    for i in xrange(n):
        str_lst = [grid[i][j].label for j in xrange(n)]
        print " " + ' '.join(str_lst)
    print


def build_grid():
    random_grid = randint(seed, size=(n, n))
    for i in xrange(n):
        for j in xrange(n):
            m = random_grid[i][j]
            if m == 0:
                grid[i][j] = Receiver([i, j])
            elif m == 1:
                grid[i][j] = Donor([i, j])
            elif m in range(2, seed):
                grid[i][j] = Empty([i, j])
    print random_grid, "\n"
    return


def is_empty(loc):
    return '_' == grid[loc[0]][loc[1]].label


def is_receiver(loc):
    return 'R' == grid[loc[0]][loc[1]].label


def is_donor(loc):
    return 'D' == grid[loc[0]][loc[1]].label


def divide(cell, empty_neighbours):
    distr = randint(1, 101)  # [1, 100]
    success = 0  # successful division = 1
    if distr < 100.0 / turn_rate:
        success = 1
        daughter_loc = random.choice(empty_neighbours)
        if 'D' == cell.label:
            grid[daughter_loc[0]][daughter_loc[1]] = Donor(daughter_loc)
            cell.maturity = floor(cell.maturity / 2)
        elif 'R' == cell.label:
            grid[daughter_loc[0]][daughter_loc[1]] = Receiver(daughter_loc)
        else:
            raise Exception("Illegal cell type")
        cell.pause = cell.refractory_div
    return success


def conjugate(cell, receiver_neighbours):
    distr = randint(1, 1001)  # [1, 1000]
    success = 0  # successful conjugation = 1
    conj_rate_rel_div_rate = expected_conj_time / expected_donor_div_time
    if distr < (1000.0 / turn_rate) / conj_rate_rel_div_rate:
        success = 1
        daughter_loc = random.choice(receiver_neighbours)
        grid[daughter_loc[0]][daughter_loc[1]] = Donor(daughter_loc)
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


def antibiotic():  # introduces antibiotic and kills donor cells
    for i in xrange(n):
        for j in xrange(n):
            loc = [i, j]
            if is_donor(loc):
                grid[i][j] = Empty(loc)
    return


def new_donor_round():  # introduce new round of donor cells
    for i in xrange(n):
        for j in xrange(n):
            loc = [i, j]
            if is_empty(loc):
                distr = randint(1, 1001)
                if distr <= 500:  # place donors in 50% of empty cells
                    grid[i][j] = Donor(loc)
    return


def run_sim(T):  # T = total sim time
    turns = int(ceil(T / time_per_turn))
    antibiotic_introduced = 0  # flag for catcing when antibiotics are introduced
    for t in xrange(turns):
        if antibiotic_introduced:
            print 'Turn ', t, ' : Time Elapsed ', t * time_per_turn, "h (New Round)"
            new_donor_round()
            antibiotic_introduced = 0
            [E, R, D] = count_cells()
            grid_data.append([t, t * time_per_turn, E, R, D])
            lattice_plotter(grid, t * time_per_turn, n, plot_lattice_folder)
            continue
        if t in turns_antibiotic:  # introduce antibiotics at turn 160 (20 h)
            print 'Turn ', t, ' : Time Elapsed ', t * time_per_turn, "h (Antibiotics Introduced)"
            antibiotic()
            antibiotic_introduced = 1
            [E, R, D] = count_cells()
            grid_data.append([t, t * time_per_turn, E, R, D])
            lattice_plotter(grid, t * time_per_turn, n, plot_lattice_folder)
            continue

        print 'Turn ', t, ' : Time Elapsed ', t * time_per_turn, "h"
        # printer()
        for i in xrange(n):
            for j in xrange(n):
                cell = grid[i][j]
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
                    else:  # divide
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

        [E, R, D] = count_cells()
        grid_data.append([t, t * time_per_turn, E, R, D])
        # print grid_data, "\n"
        lattice_plotter(grid, t * time_per_turn, n, plot_lattice_folder)
        # raw_input()

    return grid_data


# Main Function
# =================================================
def main():
    build_grid()
    run_sim(standard_run_time)

    data_name = "grid_data.csv"
    data_file = data_folder + data_name
    with open(data_file, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(grid_data)

    # convert grid_data to a dictionary
    iters = [x[0] for x in grid_data]
    time = [x[1] for x in grid_data]
    E = [x[2] for x in grid_data]
    R = [x[3] for x in grid_data]
    D = [x[4] for x in grid_data]
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


"""
-instead of explicit class structure for the states, could just use DICT (may be faster)
-utilize inheritentce, have the3 subclasses
-should we incorporate cell death

Actual lattice shape - size 4x4 (n=4)
_ _ _ _
_ _ _ _
_ _ _ _
_ _ _ _

"""
