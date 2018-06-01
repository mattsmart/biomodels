import numpy as np

#from constants import
from params import Params


VALID_PRESET_LABELS = ["preset_xyz_constant", "preset_xyz_hill"]


def presets(preset_label):
    assert preset_label in VALID_PRESET_LABELS

    if preset_label == "preset_xyz_constant":
        # DYNAMICS PARAMETERS
        system = "default"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
        feedback = "constant"  # "constant", "hill", "step", "pwlinear"
        params_dict = {
            'alpha_plus': 0.2,
            'alpha_minus': 0.5,  # 0.5
            'mu': 0.001,  # 0.01
            'a': 1.0,
            'b': 0.8,
            'c': 0.95,  # 1.2
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
        feedback = "hill"  # "constant", "hill", "step", "pwlinear"
        params_dict = {
            'alpha_plus': 0.2,
            'alpha_minus': 0.5,  # 0.5
            'mu': 0.001,  # 0.01
            'a': 1.0,
            'b': 0.8,
            'c': 0.95,  # 1.2
            'N': 100.0,  # 100.0
            'v_x': 0.0,
            'v_y': 0.0,
            'v_z': 0.0,
            'mu_base': 0.0,
            'c2': 0.0,
            'v_z2': 0.0
        }
        switching_ratio = 0.5
        feedback_multiplier_inc = 4.0
        feedback_multiplier_dec = 4.0
        params = Params(params_dict, system, feedback=feedback, switching_ratio=switching_ratio,
                        feedback_multiplier_inc=feedback_multiplier_inc, feedback_multiplier_dec=feedback_multiplier_dec)

    else:
        params = None

    return params
