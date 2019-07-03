import numpy as np

#from constants import
from params import Params


VALID_PRESET_LABELS = ["preset_xyz_constant", "preset_xyz_constant_fast", "preset_xyz_hillorig",
                       "preset_xyz_hill", "preset_xyz_hill_onlyinc", "preset_xyz_hill_onlydec",
                       "preset_xyz_tanh", "preset_xyz_tanh_onlyinc", "preset_xyz_tanh_onlydec",
                       "valley_2hit"]


def presets(preset_label):
    assert preset_label in VALID_PRESET_LABELS

    if preset_label == "preset_xyz_constant":
        # DYNAMICS PARAMETERS
        system = "default"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
        feedback = "constant"  # "constant", "hill", "step", "pwlinear", "tanh"
        params_dict = {
            'alpha_plus': 0.2,
            'alpha_minus': 1.0,  # 0.5
            'mu': 0.1,  # 0.01
            'a': 1.0,
            'b': 0.8,  # 0.8 and 1.2
            'c': 0.9,  # 0.9 and 1.1
            'N': 100.0,  # 100.0
            'v_x': 0.0,
            'v_y': 0.0,
            'v_z': 0.0,
            'mu_base': 0.0,
            'c2': 0.0,
            'v_z2': 0.0
        }
        params = Params(params_dict, system, feedback=feedback)

    elif preset_label == "preset_xyz_constant_fast":
        # DYNAMICS PARAMETERS
        system = "default"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
        feedback = "constant"  # "constant", "hill", "step", "pwlinear", "tanh"
        params_dict = {
            'alpha_plus': 0.2,
            'alpha_minus': 0.5,  # 0.5
            'mu': 0.1,  # 0.01
            'a': 1.0,
            'b': 0.99,
            'c': 1.01,  # 1.2
            'N': 100.0,  # 100.0
            'v_x': 0.0,
            'v_y': 0.0,
            'v_z': 0.0,
            'mu_base': 0.0,
            'c2': 0.0,
            'v_z2': 0.0
        }
        params = Params(params_dict, system, feedback=feedback)

    elif preset_label == "preset_xyz_hill":
        # DYNAMICS PARAMETERS
        system = "feedback_z"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
        feedback = "hill"  # "constant", "hill", "step", "pwlinear", "tanh"
        params_dict = {
            'alpha_plus': 0.2,
            'alpha_minus': 1.0,  # 0.5
            'mu': 0.0001,  # 0.01
            'a': 1.0,
            'b': 1.2,
            'c': 1.1,  # 1.2
            'N': 100.0,  # 100.0
            'v_x': 0.0,
            'v_y': 0.0,
            'v_z': 0.0,
            'mu_base': 0.0,
            'c2': 0.0,
            'v_z2': 0.0,
            'switching_ratio': 0.5,
            'mult_inc': 4.0,
            'mult_dec': 4.0,
        }
        params = Params(params_dict, system, feedback=feedback)


    elif preset_label == "preset_xyz_hill_onlyinc":
        params = presets("preset_xyz_hill")
        params = params.mod_copy({'mult_dec': 1.0})  # setting mult params to 1.0 means no feedback

    elif preset_label == "preset_xyz_hill_onlydec":
        params = presets("preset_xyz_hill")
        params = params.mod_copy({'mult_inc': 1.0})  # setting mult params to 1.0 means no feedback

    elif preset_label == "preset_xyz_hillorig":
        params = presets("preset_xyz_hill")
        params = params.mod_copy({}, feedback='hillorig')

    elif preset_label == "preset_xyz_tanh":
        params = presets("preset_xyz_hill")
        params = params.mod_copy({}, feedback='tanh')

    elif preset_label == "preset_xyz_tanh_onlyinc":
        params = presets("preset_xyz_tanh")
        params = params.mod_copy({'mult_dec': 1.0})  # setting mult params to 1.0 means no feedback

    elif preset_label == "preset_xyz_tanh_onlydec":
        params = presets("preset_xyz_tanh")
        params = params.mod_copy({'mult_inc': 1.0})  # setting mult params to 1.0 means no feedback

    elif preset_label == "valley_2hit":
        # param comparison: Fig 5a fisher 2009 theor pop bio
        # they use ensemble of 500 runs for each data point
        mu_0 = 1e-5         # their rate x->y
        mu_1 = 1e-4         # their rate y->z
        delta_1 = 2 * 1e-4  # their fitness hit to y state (i.e. b = 1 - delta_1)
        delta_2 = -0.1      # their fitness hit to z state (i.e. c = 1 - delta_2)

        # DYNAMICS PARAMETERS
        system = "default"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
        feedback = "constant"  # "constant", "hill", "step", "pwlinear", "tanh"
        params_dict = {
            'alpha_plus': mu_0,
            'alpha_minus': 0.0,  # # reversibility of x <-> y
            'mu': mu_1,  # 0.01
            'a': 1.0,
            'b': 1.0 - delta_1,
            'c': 1.0 - delta_2,  # 1.2
            'N': 100.0,  # 100.0
            'v_x': 0.0,
            'v_y': 0.0,
            'v_z': 0.0,
            'mu_base': 0.0,
            'c2': 0.0,
            'v_z2': 0.0,
            'switching_ratio': 0.0,
            'mult_inc': 0.0,
            'mult_dec': 0.0,
        }
        params = Params(params_dict, system, feedback=feedback)

    else:
        params = None

    return params
