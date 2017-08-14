import numpy as np

from constants import OUTPUT_DIR, PARAMS_ID_INV
from formulae import read_bifurc_data, read_params
from plotting import plot_fp_curves


# parameters
plt_title = "FP Curves"
flag_show = True
flag_save = False
filedir = OUTPUT_DIR
file_params = "params.csv"
file_data = "bifurc_data.csv"

# collect data
params = read_params(filedir, file_params)
data_dict = read_bifurc_data(filedir, file_data)
x0_array = np.hstack((data_dict['x0_x'], data_dict['x0_y'], data_dict['x0_z']))
x1_array = np.hstack((data_dict['x1_x'], data_dict['x1_y'], data_dict['x1_z']))
x2_array = np.hstack((data_dict['x2_x'], data_dict['x2_y'], data_dict['x2_z']))
N = params[PARAMS_ID_INV['N']]

# plot it
plot_fp_curves(x0_array, x1_array, x2_array, N, plt_title, flag_show, flag_save)