import argparse
import numpy as np
from multiprocessing import cpu_count

from data_io import write_fpt_and_params
from firstpassage import fast_fp_times, fpt_histogram, simplex_heatmap
from params import Params
from formulae import map_init_name_to_init_cond
from presets import presets


def fpt_argparser():
    parser = argparse.ArgumentParser(description='FPT data multiprocessing script')
    parser.add_argument('-n', '--ensemble', metavar='N', type=str,
                        help='ensemble size (to divide amongst cores)', default=1040)
    parser.add_argument('-s', '--suffix', metavar='S', type=str,
                        help='output filename modifier', default="main")
    parser.add_argument('-p', '--proc', metavar='P', type=str,
                        help='number of processes to distrbute job over', default=cpu_count())
    parser.add_argument('-i', '--init_name', metavar='I', type=str,
                        help='init name to map to init cond (e.g. "z_close")', default="x_all")
    return parser.parse_args()


if __name__ == "__main__":
    args = fpt_argparser()
    ensemble = int(args.ensemble)
    num_processes = int(args.proc)
    suffix = args.suffix

    # SCRIPT PARAMETERS
    plot_flag = False

    # DYNAMICS PARAMETERS
    params = presets('preset_xyz_tanh')  # TODO generalize preset in main args
    #params = params.mod_copy({'N': 100})
    init_cond = map_init_name_to_init_cond(params, args.init_name)

    fp_times, fp_states = fast_fp_times(ensemble, init_cond, params, num_processes)
    write_fpt_and_params(fp_times, fp_states, params, filename="fpt_%s_ens%d" % (params.system, ensemble), filename_mod=suffix)
    if plot_flag:
        fpt_histogram(fp_times, params, flag_show=False, figname_mod="_%s_ens%d_%s" % (params.system, ensemble, suffix))
        simplex_heatmap(fp_times, fp_states, params, flag_show=True)
