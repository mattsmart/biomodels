import numpy as np

from singlecell_class import Cell
from singlecell_constants import NUM_STEPS
from singlecell_data_io import run_subdir_setup
from singlecell_simsetup import N, XI, CELLTYPE_ID

"""
NOTES:
- projection method seems to be behaving correctly
- TODO: test vs Fig 1E Mehta 2014
- in hopfield sim, at normal temps it jumps immediately to much more stable state and stays there
"""


def main(init_type=None, iterations=NUM_STEPS, plot_period=10):

    # IO setup
    current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup()

    # Cell setup
    if init_type is None:
        init_type = "All on"
        init_state = 1 + np.zeros(N)  # start with all genes on
    else:
        init_state = XI[:, CELLTYPE_ID[init_type]]
    singlecell = Cell(init_state, init_type)
    print singlecell

    # Simulate
    for i in xrange(iterations-1):
        print "cell steps:", singlecell.steps, " H(state) =", singlecell.get_energy()
        if singlecell.steps % plot_period == 0:
            singlecell.plot_projection(pltdir=plot_lattice_folder)
        singlecell.update_state()

    # Write
    print "Writing state to file.."
    print singlecell.get_current_state()
    singlecell.write_state(data_folder)
    print "Done"


if __name__ == '__main__':
    main()
