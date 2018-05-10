import numpy as np

from multicell_class import SpatialCell
from multicell_constants import VALID_BUILDSTRINGS
from singlecell.singlecell_functions import state_to_label
from singlecell.singlecell_simsetup import XI, CELLTYPE_ID, CELLTYPE_LABELS

# TODO: could wrap all these lattice operations into Lattice class

def build_lattice_mono(n, type_1_idx=None):
    lattice = [[0 for _ in xrange(n)] for _ in xrange(n)]  # TODO: this can be made faster as np array
    for i in xrange(n):
        for j in xrange(n):
            if type_1_idx is None:
                celltype = np.random.choice(CELLTYPE_LABELS)
                init_state = XI[:, CELLTYPE_ID[celltype]]
                lattice[i][j] = SpatialCell(init_state, "%d,%d_%s" % (i, j, celltype), [i, j])
            else:
                celltype = CELLTYPE_LABELS[type_1_idx]
                init_state = XI[:, type_1_idx]
                lattice[i][j] = SpatialCell(init_state, "%d,%d_%s" % (i, j, celltype), [i, j])
    return lattice


def build_lattice_half_half(n, type_1_idx, type_2_idx):
    lattice = [[0 for _ in xrange(n)] for _ in xrange(n)]  # TODO: this can be made faster as np array
    cellname_1 = CELLTYPE_LABELS[type_1_idx]
    cellstate_1 = XI[:, type_1_idx]
    cellname_2 = CELLTYPE_LABELS[type_2_idx]
    cellstate_2 = XI[:, type_2_idx]
    for i in xrange(n):
        for j in xrange(n):
            if j >= n/2:
                lattice[i][j] = SpatialCell(cellstate_1, "%d,%d_%s" % (i, j, cellname_1), [i, j])
            else:
                lattice[i][j] = SpatialCell(cellstate_2, "%d,%d_%s" % (i, j, cellname_2), [i, j])
    return lattice


def build_lattice_memory_sequence(n, mem_list):
    lattice = [[0 for _ in xrange(n)] for _ in xrange(n)]  # TODO: this can be made faster as np array
    idx = 0
    for i in xrange(n):
        for j in xrange(n):
            mem_idx = mem_list[idx % len(mem_list)]
            cellname = CELLTYPE_LABELS[mem_idx]
            cellstate = XI[:, mem_idx]
            lattice[i][j] = SpatialCell(cellstate, "%d,%d_%s" % (i, j, cellname), [i, j])
            idx += 1
    return lattice


def build_lattice_main(n, list_of_celltype_idx, buildstring):
    print "Building %s lattice with types %s" % (buildstring, list_of_celltype_idx)
    if buildstring == "mono":
        assert len(list_of_celltype_idx) == 1
        return build_lattice_mono(n, type_1_idx=list_of_celltype_idx[0])
    elif buildstring == "dual":
        assert len(list_of_celltype_idx) == 2
        return build_lattice_half_half(n, list_of_celltype_idx[0], list_of_celltype_idx[1])
    elif buildstring == "memory_sequence":
        return build_lattice_memory_sequence(n, list_of_celltype_idx)
    else:
        raise ValueError("buildstring arg invalid, must be one of %s" % VALID_BUILDSTRINGS)


def prep_lattice_data_dict(n, duration, list_of_celltype_idx, buildstring, data_dict):
    data_dict['memory_proj_arr'] = {}
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
    else:
        raise ValueError("buildstring arg invalid, must be one of %s" % VALID_BUILDSTRINGS)
    return data_dict


def get_cell_locations(lattice, n):
    cell_locations = []
    for i in xrange(n):
        for j in xrange(n):
            loc = (i, j)
            if isinstance(lattice[i][j], SpatialCell):
                cell_locations.append(loc)
            else:
                print "Warning: non-SpatialCell at", i,j
    return cell_locations


def printer(lattice):
    n = len(lattice)
    for i in xrange(n):
        str_lst = [lattice[i][j].label for j in xrange(n)]
        print " " + ' '.join(str_lst)
    print


def printer_labels(lattice):
    n = len(lattice)
    for i in xrange(n):
        for j in xrange(n):
            state = lattice[i][j].get_current_state()
            label = state_to_label(tuple(state))
            print label, " | ",
        print

def write_state_all_cells(lattice, data_folder):
    print "Writing states to file.."
    for i in xrange(len(lattice)):
        for j in xrange(len(lattice[0])):
            lattice[i][j].write_state(data_folder)
    print "Done"
