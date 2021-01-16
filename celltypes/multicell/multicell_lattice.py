import numpy as np
import os

from multicell.multicell_class import SpatialCell
from multicell.multicell_constants import VALID_BUILDSTRINGS
from singlecell.singlecell_functions import state_to_label

# TODO: could wrap all these lattice operations into Lattice class


def build_lattice_mono(n, simsetup, type_1_idx=None):
    lattice = [[0 for _ in range(n)] for _ in range(n)]  # TODO: this can be made faster as np array
    for i in range(n):
        for j in range(n):
            if type_1_idx is None:
                celltype = np.random.choice(simsetup['CELLTYPE_LABELS'])
                init_state = simsetup['XI'][:, simsetup['CELLTYPE_ID'][celltype]]
                lattice[i][j] = SpatialCell(init_state, "%d,%d_%s" % (i, j, celltype), [i, j], simsetup)
            else:
                celltype = simsetup['CELLTYPE_LABELS'][type_1_idx]
                init_state = simsetup['XI'][:, type_1_idx]
                lattice[i][j] = SpatialCell(init_state, "%d,%d_%s" % (i, j, celltype), [i, j], simsetup)
    return lattice


def build_lattice_half_half(n, type_1_idx, type_2_idx, simsetup):
    lattice = [[0 for _ in range(n)] for _ in range(n)]  # TODO: this can be made faster as np array
    cellname_1 = simsetup['CELLTYPE_LABELS'][type_1_idx]
    cellstate_1 = simsetup['XI'][:, type_1_idx]
    cellname_2 = simsetup['CELLTYPE_LABELS'][type_2_idx]
    cellstate_2 = simsetup['XI'][:, type_2_idx]
    for i in range(n):
        for j in range(n):
            if j >= n/2:
                lattice[i][j] = SpatialCell(cellstate_1, "%d,%d_%s" % (i, j, cellname_1), [i, j], simsetup)
            else:
                lattice[i][j] = SpatialCell(cellstate_2, "%d,%d_%s" % (i, j, cellname_2), [i, j], simsetup)
    return lattice


def build_lattice_memory_sequence(n, mem_list, simsetup):
    lattice = [[0 for _ in range(n)] for _ in range(n)]  # TODO: this can be made faster as np array
    idx = 0
    for i in range(n):
        for j in range(n):
            mem_idx = mem_list[idx % len(mem_list)]
            cellname = simsetup['CELLTYPE_LABELS'][mem_idx]
            cellstate = simsetup['XI'][:, mem_idx]
            lattice[i][j] = SpatialCell(cellstate, "%d,%d_%s" % (i, j, cellname), [i, j], simsetup)
            idx += 1
    return lattice


def build_lattice_random(n, simsetup):
    lattice = [[0 for _ in range(n)] for _ in range(n)]  # TODO: this can be made faster as np array
    idx = 0
    for i in range(n):
        for j in range(n):
            cellname = str(i*j)
            cellstate = np.array([2*int(np.random.rand() < .5) - 1 for _ in range(simsetup['N'])]).T
            lattice[i][j] = SpatialCell(cellstate, "%d,%d_%s" % (i, j, cellname), [i, j], simsetup)
            idx += 1
    return lattice


def build_lattice_explicit(n, simsetup, state=None):
    assert len(state) == n**2 * simsetup['N']
    lattice = [[0 for _ in range(n)] for _ in range(n)]  # TODO: this can be made faster as np array    idx = 0
    for i in range(n):
        for j in range(n):
            cellname = str(i*j)
            posn = n * i + j
            start_spin = posn * simsetup['N']
            end_spin = (posn + 1) * simsetup['N']
            cellstate = state[start_spin:end_spin].T
            lattice[i][j] = SpatialCell(cellstate, "%d,%d_%s" % (i, j, cellname), [i, j], simsetup)
    return lattice


def build_lattice_main(n, list_of_celltype_idx, buildstring, simsetup, state=None):
    print("Building %s lattice with types %s" % (buildstring, list_of_celltype_idx))
    if buildstring == "mono":
        assert len(list_of_celltype_idx) == 1
        return build_lattice_mono(n, simsetup, type_1_idx=list_of_celltype_idx[0])
    elif buildstring == "dual":
        assert len(list_of_celltype_idx) == 2
        return build_lattice_half_half(n, list_of_celltype_idx[0], list_of_celltype_idx[1], simsetup)
    elif buildstring == "memory_sequence":
        return build_lattice_memory_sequence(n, list_of_celltype_idx, simsetup)
    elif buildstring == "random":
        return build_lattice_random(n, simsetup)
    elif buildstring == "explicit":
        return build_lattice_explicit(n, simsetup, state=state)
    else:
        raise ValueError("buildstring arg invalid, must be one of %s" % VALID_BUILDSTRINGS)


def prep_lattice_data_dict(n, duration, list_of_celltype_idx, buildstring, data_dict):
    data_dict['memory_proj_arr'] = {}
    data_dict['lattice_energy'] = np.zeros((duration, 4))        # stores: H_multi, H_self, H_app, H_pairwise_scaled
    data_dict['compressibility_full'] = np.zeros((duration, 3))  # stores: ratio, eta, eta0
    if buildstring == "mono":
        for idx in list_of_celltype_idx:
            data_dict['memory_proj_arr'][idx] = np.zeros((n*n, duration))
    elif buildstring == "dual":
        for idx in list_of_celltype_idx:
            data_dict['memory_proj_arr'][idx] = np.zeros((n*n, duration))
    elif buildstring == "memory_sequence":
        # TODO
        for idx in list_of_celltype_idx:
            data_dict['memory_proj_arr'][idx] = np.zeros((n*n, duration))
    elif buildstring == "random":
        # TODO
        for idx in list_of_celltype_idx:
            data_dict['memory_proj_arr'][idx] = np.zeros((n * n, duration))
    else:
        raise ValueError("buildstring %s invalid, must be one of %s" % (buildstring, VALID_BUILDSTRINGS))
    return data_dict


def get_cell_locations(lattice, n):
    cell_locations = []
    for i in range(n):
        for j in range(n):
            loc = (i, j)
            if isinstance(lattice[i][j], SpatialCell):
                cell_locations.append(loc)
            else:
                print("Warning: non-SpatialCell at", i,j)
    return cell_locations


def printer(lattice):
    n = len(lattice)
    for i in range(n):
        str_lst = [lattice[i][j].label for j in range(n)]
        print(" " + ' '.join(str_lst))
    print()


def printer_labels(lattice):
    n = len(lattice)
    for i in range(n):
        for j in range(n):
            state = lattice[i][j].get_current_state()
            label = state_to_label(tuple(state))
            print(label, " | ",)
        print()


def write_state_all_cells(lattice, data_folder):
    print("Writing states to file..")
    for i in range(len(lattice)):
        for j in range(len(lattice[0])):
            lattice[i][j].write_state(data_folder)
    print("Done")


def write_grid_state_int(grid_state_int, data_folder):
    """
    For each timestep, writes the n x n grid of integer states
    """
    num_steps = grid_state_int.shape[-1]
    for i in range (num_steps):
        filename = data_folder + os.sep + 'grid_state_int_at_step_%d.txt' % i
        np.savetxt(filename, grid_state_int[:, :, i], fmt='%d', delimiter=',')


def read_grid_state_int(fname):
    """
    Reads the n x n grid of integer states (for a single timestep)
    """
    return np.loadtxt(fname, dtype='int', delimiter=',')
