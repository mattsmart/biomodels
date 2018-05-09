import numpy as np

from singlecell_class import Cell
from singlecell_constants import NUM_STEPS, BURST_ERROR_PERIOD, APP_FIELD_STRENGTH
from singlecell_data_io import run_subdir_setup
from singlecell_simsetup import N, XI, CELLTYPE_ID

BETA = 2.0


def build_model(model_name="A"):

    intxn_weak = 0.5
    intxn_medium = 1.5
    applied_field_strength = 4.0

    if model_name == "A":
        spin_labels = {0: "bound_dimeric_receptor",
                       1: "pSTAT",
                       2: "SOCS",
                       3: "cytokine"}
        # effect of all on "bound_dimeric_receptor"
        J_1_on_0 = 0.0
        J_2_on_0 = intxn_medium  # ON SOCS => OFF bound_dimeric_receptor
        J_3_on_0 = intxn_weak  # ON cytokine => ON bound receptor
        # effect of each on "pSTAT"
        J_0_on_1 = intxn_medium  # ON bound_dimeric_receptor => ON pSTAT
        J_2_on_1 = 0.0
        J_3_on_1 = 0.0
        # effect of each on "SOCS"
        J_0_on_2 = 0.0
        J_1_on_2 = intxn_medium  # ON pSTAT => ON SOCS
        J_3_on_2 = 0.0
        # effect of each on "cytokine"
        J_0_on_3 = 0.0
        J_1_on_3 = intxn_medium  # ON pSTAT => ON cytokine
        J_2_on_3 = 0.0
        # fill in interaction matrix J
        intxn_matrix = np.array([[0.0,      J_1_on_0, J_2_on_0, J_3_on_0],
                                 [J_0_on_1, 0.0,      J_2_on_1, J_3_on_1],
                                 [J_0_on_2, J_1_on_2, 0.0,      J_3_on_2],
                                 [J_0_on_3, J_1_on_3, J_2_on_3, 0.0]])
        init_cond = np.array([-1, -1, -1, -1])  # all off to start
        applied_field = applied_field_strength * np.array([1, 0, 0, 0])

    else:
        print "Warning: invalid model name specified"
        spin_labels = None
        intxn_matrix = None
        applied_field = None
        init_cond = None
    return spin_labels, intxn_matrix, applied_field, init_cond
