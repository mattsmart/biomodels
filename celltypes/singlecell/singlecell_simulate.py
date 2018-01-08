import numpy as np

from singlecell_class import Cell
from singlecell_constants import NUM_STEPS, BURST_ERROR_PERIOD
from singlecell_data_io import run_subdir_setup
from singlecell_simsetup import N, XI, CELLTYPE_ID

"""
NOTES:
- projection method seems to be behaving correctly
- TODO: test vs Fig 1E Mehta 2014
- in hopfield sim, at normal temps it jumps immediately to much more stable state and stays there
"""


def main(init_state=None, init_id=None, iterations=NUM_STEPS, flag_burst_error=False, flag_write=True, analysis_subdir=None, plot_period=10):
    # TODO: if dirs is None then do run subdir setup (just current run dir?)
    # IO setup
    if analysis_subdir is None:
        current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup()
    else:
        current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup(run_subfolder=analysis_subdir)

    # Cell setup
    if init_state is None:
        if init_id is None:
            init_id = "All_on"
            init_state = 1 + np.zeros(N)  # start with all genes on
        else:
            init_state = XI[:, CELLTYPE_ID[init_id]]
    singlecell = Cell(init_state, init_id)

    # Simulate
    for i in xrange(iterations-1):
        print "cell steps:", singlecell.steps, " H(state) =", singlecell.get_energy()
        if flag_burst_error and i % BURST_ERROR_PERIOD == 0:
            singlecell.apply_burst_errors()
        if singlecell.steps % plot_period == 0:
            fig, ax, proj = singlecell.plot_projection(use_radar=False, pltdir=plot_lattice_folder)
        singlecell.update_state()

    # Write
    print "Writing state to file.."
    print singlecell.get_current_state()
    if flag_write:
        singlecell.write_state(data_folder)
    print "Done"
    return singlecell.get_state_array(), current_run_folder, data_folder, plot_lattice_folder, plot_data_folder


if __name__ == '__main__':
    main(plot_period=10)
