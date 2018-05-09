import numpy as np

from cytokine_settings import build_model
from singlecell_class import Cell
from singlecell_data_io import run_subdir_setup
from singlecell_simulate import singlecell_sim

def cytokine_sim():
    spin_labels, intxn_matrix, applied_field, init_cond = build_model()
    cell = Cell()
    singlecell_sim()
    return


if __name__ == '__main__':
    cytokine_sim()
