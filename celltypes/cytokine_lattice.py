import numpy as np

from multicell_class import SpatialCell
from singlecell.singlecell_functions import state_to_label
from singlecell.cytokine_settings import build_intercell_model, DEFAULT_CYTOKINE_MODEL


def build_cytokine_lattice_mono(n, model_name=DEFAULT_CYTOKINE_MODEL):
    spin_labels, intxn_matrix, applied_field_const, init_state, signal_matrix = build_intercell_model(model_name=DEFAULT_CYTOKINE_MODEL)
    lattice = [[0 for _ in xrange(n)] for _ in xrange(n)]  # TODO: this can be made faster as np array
    for i in xrange(n):
        for j in xrange(n):
            label = "%d,%d_%d" % (i, j, state_to_label(init_state))
            lattice[i][j] = SpatialCell(init_state, label, [i, j], memories_list=[], gene_list=spin_labels)
    return lattice, spin_labels, intxn_matrix, applied_field_const, init_state, signal_matrix


"""
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
"""
