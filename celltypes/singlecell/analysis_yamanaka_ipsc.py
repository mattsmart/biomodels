"""
Yamanaka factor names in mehta datafile:
    Sox2, Pou5f1 (oct3/4), Klf4, Mycbp, nanog
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell_constants import RUNS_FOLDER, IPSC_CORE_GENES, FLAG_PRUNE_INTXN_MATRIX, FLAG_BURST_ERRORS
from singlecell_functions import state_burst_errors, state_memory_projection_single, construct_app_field_from_genes
from singlecell_simsetup import N, P, XI, CELLTYPE_ID, A_INV, J, GENE_ID, GENE_LABELS, CELLTYPE_LABELS
from singlecell_simulate import singlecell_sim


# TODO: rename function, very similar to other gen projection timeseries function... modularize


def gen_projection_timeseries(figname='figure1_20.png'):
    FLAG_BURST_ERRORS = True
    analysis_subdir = "yamanaka"
    esc_label = 'esc'
    esc_idx = CELLTYPE_ID[esc_label]
    num_steps = 100
    app_field = construct_app_field_from_genes(IPSC_CORE_GENES, num_steps)
    proj_timeseries_array = np.zeros((num_steps, P))

    for idx, memory_label in enumerate(CELLTYPE_LABELS):
        print idx, memory_label
        cellstate_array, io_dict = singlecell_sim(init_id=memory_label, iterations=num_steps, app_field=app_field, app_field_strength=10.0,
                                                  flag_burst_error=FLAG_BURST_ERRORS, flag_write=False, analysis_subdir=analysis_subdir,
                                                  plot_period=num_steps*2)
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
