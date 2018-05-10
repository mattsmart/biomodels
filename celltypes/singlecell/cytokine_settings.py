import numpy as np


# model settings
DEFAULT_MODEL = "A"
VALID_MODELS = ["A"]

# parameter values
APP_FIELD_STRENGTH = 4.0
INTXN_WEAK = 0.5
INTXN_MEDIUM = 1.5

# io settings
RUNS_SUBDIR_CYTOKINES = "cytokines"


def build_model(model_name=DEFAULT_MODEL):

    if model_name == "A":
        spin_labels = ["bound_dimeric_receptor",
                       "pSTAT",
                       "SOCS",
                       "cytokine"]
        # effect of all on "bound_dimeric_receptor"
        J_1_on_0 = 0.0
        J_2_on_0 = -1 * INTXN_MEDIUM  # ON SOCS => OFF bound_dimeric_receptor
        J_3_on_0 = INTXN_WEAK  # ON cytokine => ON bound receptor
        # effect of each on "pSTAT"
        J_0_on_1 = INTXN_MEDIUM  # ON bound_dimeric_receptor => ON pSTAT
        J_2_on_1 = 0.0
        J_3_on_1 = 0.0
        # effect of each on "SOCS"
        J_0_on_2 = 0.0
        J_1_on_2 = INTXN_MEDIUM  # ON pSTAT => ON SOCS
        J_3_on_2 = 0.0
        # effect of each on "cytokine"
        J_0_on_3 = 0.0
        J_1_on_3 = INTXN_MEDIUM  # ON pSTAT => ON cytokine
        J_2_on_3 = 0.0
        # fill in interaction matrix J
        intxn_matrix = np.array([[0.0,      J_1_on_0, J_2_on_0, J_3_on_0],
                                 [J_0_on_1, 0.0,      J_2_on_1, J_3_on_1],
                                 [J_0_on_2, J_1_on_2, 0.0,      J_3_on_2],
                                 [J_0_on_3, J_1_on_3, J_2_on_3, 0.0]])
        init_state = np.array([-1, -1, -1, -1])  # all off to start
        applied_field_const = np.array([1, 0, 0, 0])

    else:
        print "Warning: invalid model name specified"
        spin_labels = None
        intxn_matrix = None
        applied_field_const = None
        init_state = None

    return spin_labels, intxn_matrix, applied_field_const, init_state
