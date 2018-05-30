import matplotlib.pyplot as plt
import numpy as np
import random
from operator import itemgetter
from os import sep


from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, NUM_TRAJ, TIME_START, TIME_END, NUM_STEPS, SIM_METHOD
from formulae import bifurc_value, fp_from_timeseries, get_physical_and_stable_fp
from params import Params
from plotting import plot_trajectory_mono, plot_endpoint_mono, plot_simplex, plot_trajectory
from trajectory import trajectory_simulate






if __name__ == "__main__":
    # SCRIPT PARAMETERS




    phase_portrait(params, num_traj=280, show_flag=True, basins_flag=True)
