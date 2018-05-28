import numpy as np
from os import sep

from constants import OUTPUT_DIR, PARAMS_ID_INV
from data_io import read_bifurc_data, read_params
from params import Params
from plotting import plot_fp_curves_simple


# parameters
plt_title = "FP Curves"
flag_show = True
flag_save = False
filedir = OUTPUT_DIR
file_params = "fpt_stats_N10k_c85_n64_6hr_params.csv"
file_data = "bifurc_data.csv"

# collect data
params = Params.read(filedir, file_params)
data_dict = read_bifurc_data(filedir, file_data)
x0_array = np.hstack((data_dict['x0_x'], data_dict['x0_y'], data_dict['x0_z']))
x1_array = np.hstack((data_dict['x1_x'], data_dict['x1_y'], data_dict['x1_z']))
x2_array = np.hstack((data_dict['x2_x'], data_dict['x2_y'], data_dict['x2_z']))
N = params.N

print N

# plot it
plot_fp_curves_simple(x0_array, data_dict['x0_stab'], x1_array, data_dict['x1_stab'], x2_array, data_dict['x2_stab'], N,
                      plt_title, flag_show, flag_save, colourbinary=True)
