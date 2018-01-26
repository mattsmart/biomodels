"""
Yamanaka factor names in mehta datafile:
    Sox2, Pou5f1 (oct3/4), Klf4, Mycbp, nanog
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell_constants import RUNS_FOLDER, IPSC_CORE_GENES
from singlecell_functions import state_burst_errors, state_memory_projection_single
from singlecell_simsetup import N, P, XI, CELLTYPE_ID, A_INV, J, GENE_ID, GENE_LABELS, CELLTYPE_LABELS
from singlecell_simulate import main


# TODO: rename function, very similar to other gen projection timeseries function... modularize

def get_memory_proj_timeseries(state_array, memory_idx):
    num_steps = np.shape(state_array)[1]
    timeseries = np.zeros(num_steps)
    for time_idx in xrange(num_steps):
        timeseries[time_idx] = state_memory_projection_single(state_array, time_idx, memory_idx)
    return timeseries


def construct_app_field_from_genes(gene_list, num_steps):
    app_field = np.zeros((N, num_steps))
    for label in gene_list:
        app_field[GENE_ID[label], :] += 1
        print app_field[GENE_ID[label]-1:GENE_ID[label]+2,0:5]
    return app_field


def gen_projection_timeseries(figname='figure1_20.png'):
    FLAG_BURST_ERRORS = True
    FLAG_DILUTE_INTXNS = False  # TODO: propogate through to singlecell_functions or singlecell_simsetup?
    analysis_subdir = "yamanaka"
    esc_label = 'esc'
    esc_idx = CELLTYPE_ID[esc_label]
    num_steps = 100
    app_field = construct_app_field_from_genes(IPSC_CORE_GENES, 100)
    ratio_amounts = np.linspace(0.0, 0.5, 50)  # this makes more sense and plot makes more sense
    proj_timeseries_array = np.zeros((num_steps, P))

    for idx, memory_label in enumerate(CELLTYPE_LABELS):
        print idx, memory_label
        cellstate_array, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = main(init_id=memory_label, iterations=num_steps, app_field=app_field ,flag_burst_error=FLAG_BURST_ERRORS,
                                                                                                       flag_write=False, analysis_subdir=analysis_subdir, plot_period=num_steps*2)
        proj_timeseries_array[:, idx] = get_memory_proj_timeseries(cellstate_array, esc_idx)[:]

    # cleanup output folders from main()
    # TODO: function to cleanup runs analysis subdir call here call

    # plot output
    plt.plot(xrange(num_steps), proj_timeseries_array, color='blue', linewidth=0.75)
    plt.title('Test %s projection after yamanaka field for all memories' % esc_label)
    plt.ylabel('proj on memory %s' % (esc_label))
    plt.xlabel('Time (10^3 updates, all spins)')
    plt.savefig(RUNS_FOLDER + analysis_subdir + os.sep + figname)
    print N


if __name__ == '__main__':
    gen_projection_timeseries(figname='figure1_20.png')
