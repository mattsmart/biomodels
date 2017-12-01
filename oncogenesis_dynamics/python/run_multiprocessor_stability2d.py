import numpy as np
import time
from multiprocessing import Pool
from multiprocessing import cpu_count

from constants import OUTPUT_DIR
from data_io import write_matrix_data_and_idx_vals
from stability_diagram import plot_stable_fp_count_2d, get_stable_fp_count_2d, get_gap_data_2d, plot_gap_data_2d

# CONSTANTS
NUM_PROCESSES = -1 + cpu_count()

# PARAMS
alpha_plus = 0.2  # 0.05 #0.4
alpha_minus = 0.5  # 4.95 #0.5
mu = 0.001  # 0.01
a = 1.0
b = 0.8
c = 0.85  # 2.6 #1.2
N = 100.0  # 100
v_x = 0.0
v_y = 0.0
v_z = 0.0
mu_base = 0.0
params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]
ode_system = "feedback_z"

# ARGS TO PASS
param_1_name = "b"
param_1_start = 0.5 #0.975 #0.5
param_1_stop = 1.1 #1.005 #1.1
param_1_steps = 210 #350 #700
param_1_range = np.linspace(param_1_start, param_1_stop, param_1_steps)
param_2_name = "c"
param_2_start = 0.6 #0.6  # 1.1 #0.7
param_2_stop = 1.1 #1.005 #1.1  # 1.3 #0.95
param_2_steps = 100 #250 #500
param_2_range = np.linspace(param_2_start, param_2_stop, param_2_steps)

#param_name_to_split = 2
#assert param_name_to_split in [1,2]

data_id = "fpcount"  # gapdist or fpcount
flag_physicalfp = True
flag_plot = True
if data_id == "fpcount":
    data_fnstr = "phys"*flag_physicalfp + "fpcount2d_full"
    data_fn = get_stable_fp_count_2d
    plot_fn = plot_stable_fp_count_2d
    kwargs_dict = {'figname_mod': None, 'flag_phys': flag_physicalfp}
    kwargs_plot_dict = {'figname_mod': "sum", 'flag_show': True, 'flag_phys': flag_physicalfp}
elif data_id == "gapdist":
    data_fnstr = "gapdist2d_full"
    data_fn = get_gap_data_2d
    plot_fn = plot_gap_data_2d
    kwargs_dict = {'figname_mod': None}
    kwargs_plot_dict = {'figname_mod': "sum", 'flag_show': True}
else:
   print "ERROR: %s invalid" % data_id

pool_fn = data_fn
def pool_fn_wrapper(fn_args_dict):
    return pool_fn(*fn_args_dict['args'], **fn_args_dict['kwargs'])

if __name__ == "__main__":
    fn_args_dict = [0]*NUM_PROCESSES
    print "NUM_PROCESSES:", NUM_PROCESSES
    assert param_1_steps % NUM_PROCESSES == 0  # gonna make a picture don't want gaps
    for i in xrange(NUM_PROCESSES):
        range_step = param_1_steps / NUM_PROCESSES
        param_1_reduced_range = param_1_range[i*range_step : (1 + i)*range_step]
        print "process:", i, "job size:", len(param_1_reduced_range), "x", len(param_2_range)
        fn_args_dict[i] = {'args': (params, param_1_name, param_1_reduced_range, param_2_name, param_2_range, ode_system),
                           'kwargs': kwargs_dict}
    t0 = time.time()
    pool = Pool(NUM_PROCESSES)
    results = pool.map(pool_fn_wrapper, fn_args_dict)
    pool.close()
    pool.join()
    print "TIMER:", time.time() - t0

    results_dim = np.shape(results[0])
    results_collected = np.zeros((results_dim[0]*NUM_PROCESSES, results_dim[1]))
    for i, result in enumerate(results):
        results_collected[i*results_dim[0]:(i+1)*results_dim[0], :] = result
    write_matrix_data_and_idx_vals(results_collected, param_1_range, param_2_range, data_fnstr,
                                   param_1_name, param_2_name, output_dir=OUTPUT_DIR)
    if flag_plot:
        plot_fn(results_collected, params, param_1_name, param_1_range, param_2_name,
                param_2_range, ode_system, **kwargs_plot_dict)