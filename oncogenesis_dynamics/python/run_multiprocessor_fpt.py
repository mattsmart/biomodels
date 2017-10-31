import argparse
import numpy as np
from multiprocessing import cpu_count

from firstpassage import write_fpt_and_params, fast_fp_times, fpt_histogram


def fpt_argparser():
    parser = argparse.ArgumentParser(description='FPT data multiprocessing script')
    parser.add_argument('-n', '--ensemble', metavar='N', type=str,
                        help='ensemble size (to divide amongst cores)', default=1024)
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

    init_cond = [int(N), 0, 0]

    fp_times = fast_fp_times(ensemble, init_cond, params, system, num_processes)
    write_fpt_and_params(fp_times, params, system, filename="fpt_%s_ens%d" % (system, ensemble), filename_mod=suffix)
    if plot_flag:
        fpt_histogram(fp_times, params, system, show_flag=False, figname_mod="_%s_ens%d_%s" % (system, ensemble, suffix))
