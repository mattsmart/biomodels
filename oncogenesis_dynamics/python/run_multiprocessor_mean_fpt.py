import argparse
import numpy as np
import time
from multiprocessing import cpu_count

from constants import PARAMS_ID_INV
from data_io import write_varying_mean_sd_fpt_and_params, read_varying_mean_sd_fpt_and_params
from firstpassage import fast_mean_fpt_varying, plot_mean_fpt_varying


def fpt_argparser():
    parser = argparse.ArgumentParser(description='FPT data multiprocessing script')
    parser.add_argument('-p', '--proc', metavar='P', type=str,
                        help='number of processes to distrbute job over', default=cpu_count())
    parser.add_argument('-n', '--ensemble', metavar='N', type=str,
                        help='ensemble size (to divide amongst cores)', default=96)
    parser.add_argument('-s', '--suffix', metavar='S', type=str,
                        help='output filename modifier', default="")
    parser.add_argument('-c', '--param_to_vary', metavar='C', type=str,
                        help='param to vary (e.g. "c")', default="c")
    parser.add_argument('-i', '--init_name', metavar='I', type=str,
                        help='init name to map to init cond (e.g. "z_close")', default="x_all")
    parser.add_argument('-k', '--size_linspace', metavar='K', type=str,
                        help='search size for np.linspace(L_val,R_val,K))', default=100)
    parser.add_argument('-l', '--left_idx', metavar='L', type=str,
                        help='slice is np.linspace(L_val,R_val,K))[left:right]', default=None)
    parser.add_argument('-r', '--right_idx', metavar='R', type=str,
                        help='slice is np.linspace(L_val,R_val,K))[left:right]', default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = fpt_argparser()
    num_processes = int(args.proc)
    ensemble = int(args.ensemble)
    suffix = args.suffix
    param_to_vary = args.param_to_vary
    init_name = args.init_name
    size_linspace = int(args.size_linspace)
    if args.left_idx is None:
        print "args.left_idx is None, setting to %d" % 0
        pv1_idx = 0
    else:
        pv1_idx = int(args.left_idx)
    if args.right_idx is None:
        print "args.right_idx is None, setting to %d" % size_linspace
        pv2_idx = size_linspace
    else:
        pv2_idx = int(args.right_idx)

    # SCRIPT PARAMETERS
    system = "feedback_z"  # "feedback_mu_XZ_model" or "feedback_z"
    plot_flag = False

    # DYNAMICS PARAMETERS
    alpha_plus = 0.2  # 0.2
    alpha_minus = 0.5  # 0.5
    mu = 0.001  # 0.01
    a = 1.0
    b = 0.6
    c = 0.95  # 1.2
    N = 10000.0  # 100.0
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = 0.0  #mu*1e-1
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]

    if param_to_vary == 'N':
        #param_set = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,1000,2000,3000,4000,5000,10000]
        param_set = np.logspace(1.0, 4.0, num=size_linspace)
        param_set = [np.round(a) for a in param_set]
    elif param_to_vary == 'c':
        param_set = np.linspace(0.75, 0.95, num=size_linspace)
    else:
        assert param_to_vary in PARAMS_ID_INV.keys()
        print "ERROR:", param_to_vary, "not yet implemented in main"

    param_set = param_set[pv1_idx:pv2_idx]
    print param_set

    t0 = time.time()
    mean_fpt_varying, sd_fpt_varying = fast_mean_fpt_varying(param_to_vary, param_set, params, system,
                                                             num_processes, init_name=init_name, samplesize=ensemble)
    print "Elapsed time:", time.time() - t0

    datafile, paramfile = \
        write_varying_mean_sd_fpt_and_params(mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system,
                                             filename_mod=suffix)
    mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system = \
        read_varying_mean_sd_fpt_and_params(datafile, paramfile)
    if plot_flag:
        plot_mean_fpt_varying(mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system, ensemble,
                              show_flag=True, figname_mod="_%s_n%d" % (param_to_vary, ensemble))
