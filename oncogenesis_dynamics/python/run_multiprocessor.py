import numpy as np
import time
from multiprocessing import Pool
from multiprocessing import cpu_count

from stability_diagram import plot_stable_fp_count_2d, get_stable_fp_count_2d

# CONSTANTS
NUM_PROCESSES = -1 + cpu_count()

# PARAMS
alpha_plus = 0.2  # 0.05 #0.4
alpha_minus = 0.5  # 4.95 #0.5
mu = 0.001  # 0.01
a = 1.0
b = 0.8
c = 0.6  # 2.6 #1.2
N = 100.0  # 100
v_x = 0.0
v_y = 0.0
v_z = 0.0
params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z]
ode_system = "feedback_z"

# ARGS TO PASS
param_1_name = "b"
param_1_start = 0.5
param_1_stop = 1.2
param_1_steps = 700
param_1_range = np.linspace(param_1_start, param_1_stop, param_1_steps)
param_2_name = "c"
param_2_start = 0.8  # 1.1 #0.7
param_2_stop = 1.2  # 1.3 #0.95
param_2_steps = 400
param_2_range = np.linspace(param_2_start, param_2_stop, param_2_steps)

pool_fn = get_stable_fp_count_2d  #plot_stable_fp_count_2d
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
                           'kwargs': {'figname_mod': None}}
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
    plot_stable_fp_count_2d(results_collected, params, param_1_name, param_1_range, param_2_name,
                           param_2_range, ode_system, figname_mod="sum")
