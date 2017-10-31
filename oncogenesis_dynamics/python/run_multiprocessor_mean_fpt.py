import argparse
import numpy as np
import time
from multiprocessing import cpu_count

from firstpassage import fast_mean_fpt_varying, plot_mean_fpt_varying, write_varying_mean_sd_fpt_and_params, read_varying_mean_sd_fpt_and_params


def fpt_argparser():
    parser = argparse.ArgumentParser(description='FPT data multiprocessing script')
    parser.add_argument('-n', '--ensemble', metavar='N', type=str,
                        help='ensemble size (to divide amongst cores)', default=32)
    parser.add_argument('-s', '--suffix', metavar='S', type=str,
                        help='output filename modifier', default="main")
    parser.add_argument('-p', '--proc', metavar='P', type=str,
                        help='number of processes to distrbute job over', default=cpu_count())
    return parser.parse_args()


if __name__ == "__main__":
    args = fpt_argparser()
    ensemble = int(args.ensemble)
    num_processes = int(args.proc)
    suffix = args.suffix

    # SCRIPT PARAMETERS
    system = "feedback_z"  # "feedback_mu_XZ_model" or "feedback_z"
    plot_flag = False

    # DYNAMICS PARAMETERS
    alpha_plus = 0.2  # 0.2
    alpha_minus = 0.5  # 0.5
    mu = 0.001  # 0.01
    a = 1.0
    b = 0.8
    c = 0.95  # 1.2
    N = 10000.0  # 100.0
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = 0.0  #mu*1e-1
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]

    param_to_vary = 'c'
    #param_set = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,1000,2000,3000,4000,5000,10000]
    #param_set = np.logspace(1.0, 3.0, num=30)
    param_set = np.linspace(0.75, 0.95, num=30)

    t0 = time.time()
    mean_fpt_varying, sd_fpt_varying = fast_mean_fpt_varying(param_to_vary, param_set, params, system,
                                                             num_processes, samplesize=ensemble)
    print "Elapsed time:", time.time() - t0

    datafile, paramfile = \
        write_varying_mean_sd_fpt_and_params(mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system)
    mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system = \
        read_varying_mean_sd_fpt_and_params(datafile, paramfile)
    if plot_flag:
        plot_mean_fpt_varying(mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system, ensemble,
                              show_flag=True, figname_mod="_%s_n%d" % (param_to_vary, ensemble))
