import datetime
import numpy as np
import os

from singlecell_constants import RUNS_FOLDER, SETTINGS_FILE


def run_subdir_setup(run_subfolder=None):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %I.%M.%S%p")
    time_folder = current_time
    if run_subfolder is None:
        current_run_folder = RUNS_FOLDER + os.sep + time_folder
    else:
        current_run_folder = RUNS_FOLDER + os.sep + run_subfolder + os.sep + time_folder

    # make subfolders in the timestamped run directory:
    data_folder = os.path.join(current_run_folder, "data")
    plot_lattice_folder = os.path.join(current_run_folder, "plot_lattice")
    plot_data_folder = os.path.join(current_run_folder, "plot_data")
    dir_list = [RUNS_FOLDER, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder]
    for dirs in dir_list:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    # io path storage to pass around
    io_dict = {'basedir': current_run_folder,
               'datadir': data_folder,
               'latticedir': plot_lattice_folder,
               'plotdir': plot_data_folder,
               'runinfo': current_run_folder + os.sep + SETTINGS_FILE}

    # make base settings file with first line as the base output dir
    runinfo_append(io_dict, ('basedir', current_run_folder))

    #return current_run_folder, data_folder, plot_lattice_folder, plot_data_folder
    return io_dict


def state_write(state, row_vals, col_vals, dataname, rowname, colname, output_dir):
    # here row refers to time array and col refers to gene labels (ie. name for ith element of state vector)
    datapath = output_dir + os.sep + dataname + ".txt"
    rowpath = output_dir + os.sep + dataname + '_' + rowname + ".txt"
    colpath = output_dir + os.sep + dataname + '_' + colname + ".txt"
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


def runinfo_init(io_dict):
    # TODO implement settings file initialization
    """
    info_list = ..........
    with open(io_dict['runinfo'], 'a') as runinfo:
        runinfo.write(','.join(str(s) for s in info_list) + '\n')
    """
    return



def runinfo_append(io_dict, info_list, lol=False):
    # lol: list of list flag
    if lol:
        with open(io_dict['runinfo'], 'a') as runinfo:
            for line in info_list:
                runinfo.write(','.join(str(s) for s in line) + '\n')
    else:
        with open(io_dict['runinfo'], 'a') as runinfo:
            runinfo.write(','.join(str(s) for s in info_list) + '\n')
    return
