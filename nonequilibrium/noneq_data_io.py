import datetime
import numpy as np
import os
from os import sep

from noneq_settings import RUNS_FOLDER


def run_subdir_setup(run_subfolder=None):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %I.%M.%S%p")
    time_folder = current_time + os.sep
    if run_subfolder is None:
        current_run_folder = RUNS_FOLDER + time_folder
    else:
        current_run_folder = RUNS_FOLDER + run_subfolder + os.sep + time_folder

    # subfolders in the timestamped run directory:
    data_folder = os.path.join(current_run_folder, "data")
    plot_lattice_folder = os.path.join(current_run_folder, "plot_lattice")
    plot_data_folder = os.path.join(current_run_folder, "plot_data")

    dir_list = [RUNS_FOLDER, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder]
    for dirs in dir_list:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    return current_run_folder, data_folder, plot_lattice_folder, plot_data_folder


def state_write(state, row_vals, col_vals, dataname, rowname, colname, output_dir):
    # here row refers to time array and col refers to gene labels (ie. name for ith element of state vector)
    datapath = output_dir + sep + dataname + ".txt"
    rowpath = output_dir + sep + dataname + '_' + rowname + ".txt"
    colpath = output_dir + sep + dataname + '_' + colname + ".txt"
    np.savetxt(datapath, np.array(state), delimiter=",", fmt="%d")
    np.savetxt(rowpath, np.array(row_vals), delimiter=",")
    np.savetxt(colpath, np.array(col_vals), delimiter=",", fmt="%s")
    return datapath, rowpath, colpath


def state_read(datapath, rowpath, colpath):
    # here row refers to time array and col refers to gene labels (ie. name for ith element of state vector)
    state = np.loadtxt(datapath, delimiter=",")
    row = np.loadtxt(rowpath, delimiter=",", dtype=float)
    col = np.loadtxt(colpath, delimiter=",", dtype=str)
    return state, row, col
