import argparse
import numpy as np
from multiprocessing import cpu_count

from data_io import write_fpt_and_params
from firstpassage import fast_fp_times, fpt_histogram
from params import Params


def fpt_argparser():
    parser = argparse.ArgumentParser(description='FPT data multiprocessing script')
    parser.add_argument('-n', '--ensemble', metavar='N', type=str,
                        help='ensemble size (to divide amongst cores)', default=1040)
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
    system = "feedback_z"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
    feedback = "hill"      # "constant", "hill", "step", "pwlinear"
    plot_flag = False

    # DYNAMICS PARAMETERS
    params_dict = {
        'alpha_plus': 0.2,
        'alpha_minus': 0.5,  # 0.5
        'mu': 0.001,  # 0.01
        'a': 1.0,
        'b': 0.8,
        'c': 0.95,  # 1.2
        'N': 10000.0,  # 100.0
        'v_x': 0.0,
        'v_y': 0.0,
        'v_z': 0.0,
        'mu_base': 0.0,
        'c2': 0.0,
        'v_z2': 0.0
    }
    params = Params(params_dict, system, feedback=feedback)

    init_cond = np.zeros(params.numstates, dtype=int)
    init_cond[0] = int(params.N)

    fp_times = fast_fp_times(ensemble, init_cond, params, num_processes)
    write_fpt_and_params(fp_times, params, filename="fpt_%s_ens%d" % (params.system, ensemble), filename_mod=suffix)
    if plot_flag:
        fpt_histogram(fp_times, params, flag_show=False, figname_mod="_%s_ens%d_%s" % (params.system, ensemble, suffix))
