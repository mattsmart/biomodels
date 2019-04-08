import matplotlib.pyplot as plt
import numpy as np
import os

from constants import OUTPUT_DIR
from data_io import read_matrix_data_and_idx_vals, read_params, read_fpt_and_params


def subsample_data():
    # TODO
    return fpt_data_subsampled


def figure_fpt_multihist(param_1_name, param_2_name, figname_mod="", flag_show=True,
                         outdir=OUTPUT_DIR):
    fs = 16
    # TODO
    plt.savefig(outdir + os.sep + 'fig_%s_%s_%s.pdf' % (param_1_name, param_2_name, figname_mod), bbox_inches='tight')
    if flag_show:
        plt.show()
    return plt.gca()


if __name__ == "__main__":
    multihist = True
    basedir = "figures"

    if multihist:
        print 'generating multihist figure'
        # TODO
