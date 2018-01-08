import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell_constants import RUNS_FOLDER
from singlecell_functions import state_burst_errors, state_memory_projection_single
from singlecell_simsetup import N, XI, CELLTYPE_ID, A_INV, J
from singlecell_simulate import main


def get_memory_proj_timeseries(state_array, memory_idx):
    num_steps = np.shape(state_array)[1]
    timeseries = np.zeros(num_steps)
    for time_idx in xrange(num_steps):
        timeseries[time_idx] = state_memory_projection_single(state_array, time_idx, memory_idx)
    return timeseries


def gen_projection_timeseries(figname='figure1_20.png'):

    FLAG_BURST_ERRORS = True
    FLAG_DILUTE_INTXNS = False  # TODO: propogate through to singlecell_functions or singlecell_simsetup?
    analysis_subdir = "mehta1E"
    esc_label = 'esc'
    esc_idx = CELLTYPE_ID[esc_label]
    init_state_esc = XI[:, esc_idx]
    num_steps = 100
    ratio_amounts = np.linspace(0.0, 0.5, 50)  # this makes more sense and plot makes more sense
    #ratio_amounts = np.zeros(100) + 0.15  # thesis said to randomly flip 15% to get IC for figure 1E
    proj_timeseries_array = np.zeros((num_steps, len(ratio_amounts)))

    for idx, ratio_to_flip in enumerate(ratio_amounts):
        subsample_state = state_burst_errors(init_state_esc, ratio_to_flip=ratio_to_flip)
        cellstate_array, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = main(init_state=subsample_state, iterations=num_steps, flag_burst_error=FLAG_BURST_ERRORS,
                                                                                                       flag_write=False, analysis_subdir=analysis_subdir, plot_period=num_steps*2)
        proj_timeseries_array[:, idx] = get_memory_proj_timeseries(cellstate_array, esc_idx)[:]

    # cleanup output folders from main()
    # TODO: function to cleanup runs subdir mehta1E call here call

    # plot output
    plt.plot(xrange(num_steps), proj_timeseries_array, color='blue', linewidth=0.75)
    plt.title('Mehta Fig 1E analog: proj on memory %s for varying IC' % (esc_label))
    plt.ylabel('proj on memory %s' % (esc_label))
    plt.xlabel('Time (10^3 updates, all spins)')
    plt.savefig(RUNS_FOLDER + analysis_subdir + os.sep + figname)
    print N


gen_projection_timeseries('figure1_0_A.png')
gen_projection_timeseries('figure1_0_B.png')
gen_projection_timeseries('figure1_0_C.png')
gen_projection_timeseries('figure1_0_D.png')
gen_projection_timeseries('figure1_0_E.png')
