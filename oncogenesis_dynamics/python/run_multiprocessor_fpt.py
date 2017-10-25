import numpy as np
import time
from multiprocessing import Pool
from multiprocessing import cpu_count

from firstpassage import get_fpt, fpt_histogram, write_fpt_and_params

# CONSTANTS
NUM_PROCESSES = -1 + cpu_count()


pool_fn = get_fpt
def pool_fn_wrapper(fn_args_dict):
    if fn_args_dict['kwargs'] is not None:
        return pool_fn(*fn_args_dict['args'], **fn_args_dict['kwargs'])
    else:
        return pool_fn(*fn_args_dict['args'])


if __name__ == "__main__":

    # SCRIPT PARAMETERS
    system = "feedback_z"  # "feedback_mu_XZ_model" or "feedback_z"
    num_steps = 100000
    ensemble = 1001

    # DYNAMICS PARAMETERS
    alpha_plus = 0.2  # 0.2
    alpha_minus = 0.5  # 0.5
    mu = 0.001  # 0.01
    a = 1.0
    b = 0.8
    c = 0.81  # 1.2
    N = 100.0  # 100.0
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = 0.0  #mu*1e-1
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]

    init_cond = [int(N), 0, 0]

    fn_args_dict = [0]*NUM_PROCESSES
    print "NUM_PROCESSES:", NUM_PROCESSES
    assert ensemble % NUM_PROCESSES == 0
    for i in xrange(NUM_PROCESSES):
        subensemble = ensemble / NUM_PROCESSES
        print "process:", i, "job size:", subensemble, "runs"
        fn_args_dict[i] = {'args': (subensemble, init_cond, num_steps, params, system),
                           'kwargs': None}
    t0 = time.time()
    pool = Pool(NUM_PROCESSES)
    results = pool.map(pool_fn_wrapper, fn_args_dict)
    pool.close()
    pool.join()
    print "TIMER:", time.time() - t0

    fp_times = np.zeros(ensemble)
    for i, result in enumerate(results):
        fp_times[i*subensemble:(i+1)*subensemble] = result
    print "mean", np.mean(fp_times)
    write_fpt_and_params(fp_times, params, system, filename="fpt_xyz_feedbackz_1000")
    fpt_histogram(fp_times, params, system, show_flag=True, figname_mod="_xyz_feedbackz")
