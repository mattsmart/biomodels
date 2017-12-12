import datetime
import numpy as np
import os
import re
from os import sep

from singlecell_constants import ZSCORE_DATAFILE, RUNS_FOLDER


def run_subdir_setup():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %I.%M.%S%p")
    time_folder = current_time + os.sep
    current_run_folder = RUNS_FOLDER + time_folder

    # subfolders in the timestamped run directory:
    data_folder = os.path.join(current_run_folder, "data")
    plot_lattice_folder = os.path.join(current_run_folder, "plot_lattice")
    plot_data_folder = os.path.join(current_run_folder, "plot_data")

    dir_list = [RUNS_FOLDER, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder]
    for dirs in dir_list:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    return current_run_folder, data_folder, plot_lattice_folder, plot_data_folder


def load_singlecell_data(zscore_datafile=ZSCORE_DATAFILE):
    """
    Returns list of cell types (size p), list of TFs (size N), and xi array where xi_ij is ith TF value in cell type j
    Note the Mehta SI file has odd formatting, array text file is read in as single line:
    http://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1003734.s005&type=supplementary
    """
    gene_labels = []
    with open(zscore_datafile) as f:
        origline = f.readline()
        filelines = origline.split('\r')
        for idx_row, row in enumerate(filelines):
            row_split = re.split(r'\t', row)
            if idx_row == 0:  # celltypes line, first row
                celltype_labels = row_split[1:]
            else:
                gene_labels.append(row_split[0])
    # reloop to get data without excessive append calls
    expression_data = np.zeros((len(gene_labels), len(celltype_labels)))
    with open(zscore_datafile) as f:
        origline = f.readline()
        filelines_dataonly = origline.split('\r')[1:]
        for idx_row, row in enumerate(filelines_dataonly):
            row_split_dataonly = re.split(r'\t', row)[1:]
            expression_data[idx_row,:] = [float(val) for val in row_split_dataonly]
    return expression_data, celltype_labels, gene_labels


def binarize_data(xi):
    return np.where(xi > 0, 1, -1)


def state_write(state, row_vals, col_vals, dataname, rowname, colname, output_dir):
    # here row refers to time array and col refers to gene labels (ie. name for ith element of state vector)
    datapath = output_dir + sep + dataname + ".txt"
    rowpath = output_dir + sep + dataname + '_' + rowname + ".txt"
    colpath = output_dir + sep + dataname + '_' + colname + ".txt"
    np.savetxt(datapath, np.array(state), delimiter=",")
    np.savetxt(rowpath, np.array(row_vals), delimiter=",")
    np.savetxt(colpath, np.array(col_vals), delimiter=",")
    return datapath, rowpath, colpath


def state_read(datapath, rowpath, colpath):
    # here row refers to time array and col refers to gene labels (ie. name for ith element of state vector)
    state = np.loadtxt(datapath, delimiter=",", dtype=float)
    row = np.loadtxt(rowpath, delimiter=",", dtype=float)
    col = np.loadtxt(colpath, delimiter=",", dtype=str)
    return state, row, col


if __name__ == '__main__':
    xi, celltype_labels, gene_labels = load_singlecell_data(ZSCORE_DATAFILE)
    xi_bool = binarize_data(xi)
