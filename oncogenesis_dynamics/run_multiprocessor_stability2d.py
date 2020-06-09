import numpy as np
import time
from multiprocessing import Pool
from multiprocessing import cpu_count

from constants import OUTPUT_DIR
from data_io import write_matrix_data_and_idx_vals
from params import Params
from presets import presets
from stability_diagram import plot_fp_count_2d, get_fp_count_2d, get_gap_data_2d, plot_gap_data_2d

# CONSTANTS
NUM_PROCESSES = cpu_count() / 2  # 16

# DYNAMICS PARAMETERS
params = presets('preset_xyz_constant')
#params = presets('preset_xyz_tanh')

# ARGS TO PASS
param_1_name = "gamma"  # b
param_1_start = 1  # 0.2
param_1_stop = 6   # 2.0
param_1_steps = 1*36
param_1_range = np.linspace(param_1_start, param_1_stop, param_1_steps)
"""
param_2_name = "c"
param_2_start = 0.2  #0.01 #0.7
param_2_stop = 2.0#10.0 #0.95
param_2_steps = 1*30
param_2_range = np.linspace(param_2_start, param_2_stop, param_2_steps)
#param_2_range = np.logspace(np.log10(param_2_start), np.log10(param_2_stop), param_2_steps)
"""
param_2_name = "mu"
param_2_start = -5 #0.7
param_2_stop = 1  #0.95
param_2_steps = 1*30
param_2_range = np.logspace(param_2_start, param_2_stop, param_2_steps)

data_id = "gapdist"  # gapdist or fpcount
flag_physicalfp = True
flag_stablefp = True  # sum overrides this
flag_sumfp = False
flag_plot = True
if data_id == "fpcount":
    data_fnstr = "fp" + "Phys"*flag_physicalfp + "Stable"*flag_stablefp + "Sum"*flag_sumfp + "fpcount2d_full"
    data_fn = get_fp_count_2d
    plot_fn = plot_fp_count_2d
    kwargs_dict = {'figname_mod': None, 'flag_phys': flag_physicalfp, 'flag_stable': flag_stablefp, 'flag_sum': flag_sumfp}
    kwargs_plot_dict = {'figname_mod': "sum", 'flag_show': True, 'flag_phys': flag_physicalfp, 'flag_stable':flag_stablefp, 'flag_sum': flag_sumfp}
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
        fn_args_dict[i] = {'args': (params, param_1_name, param_1_reduced_range, param_2_name, param_2_range),
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
    params.write(OUTPUT_DIR, data_fnstr + "_params.csv")

    if flag_plot:
        plot_fn(results_collected, params, param_1_name, param_1_range, param_2_name,
                param_2_range, **kwargs_plot_dict)
