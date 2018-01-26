import numpy as np

from singlecell_class import Cell
from singlecell_constants import NUM_STEPS, BURST_ERROR_PERIOD, APP_FIELD_STRENGTH
from singlecell_data_io import run_subdir_setup
from singlecell_simsetup import N, XI, CELLTYPE_ID

"""
NOTES:
- projection method seems to be behaving correctly
- TODO: test vs Fig 1E Mehta 2014
- in hopfield sim, at normal temps it jumps immediately to much more stable state and stays there
"""


def main(init_state=None, init_id=None, iterations=NUM_STEPS, app_field=None, app_field_strength=APP_FIELD_STRENGTH, flag_burst_error=False, flag_write=True, analysis_subdir=None, plot_period=10):
    """
    init_state: N x 1
    init_id: None, or memory label like 'esc', or arbitrary label (e.g. 'All on')
    iterations: main simulation loop duration
    app_field: size N x timesteps applied field array; column k is field to apply at timestep k
    app_field_strength: scales app_field magnitude
    flag_burst_error: if True, randomly flip some TFs at each BURST_ERROR_PERIOD (see ...constants.py)
    flag_write: False only if want to avoid saving state to file
    analysis_subdir: use to store data for non-standard runs
    plot_period: period at which to plot cell state projection onto memory subspace
    """
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

    # Input checks
    if app_field is not None:
        assert len(app_field) == N
        assert len(app_field[0]) == iterations
    else:
        app_field_timestep = None

    # Simulate
    for step in xrange(iterations-1):
        print "cell steps:", singlecell.steps, " H(state) =", singlecell.get_energy()
        # apply burst errors
        if flag_burst_error and step % BURST_ERROR_PERIOD == 0:
            singlecell.apply_burst_errors()
        # prep applied field TODO see if better speed to pass array of zeros and ditch all these if not None checks...
        if app_field is not None:
            app_field_timestep = app_field[:, step]
        if singlecell.steps % plot_period == 0:
            fig, ax, proj = singlecell.plot_projection(use_radar=True, pltdir=plot_lattice_folder)
        singlecell.update_state(app_field=app_field_timestep, app_field_strength=app_field_strength)

    # Write
    print "Writing state to file.."
    print singlecell.get_current_state()
    if flag_write:
        singlecell.write_state(data_folder)
    print "Done"
    return singlecell.get_state_array(), current_run_folder, data_folder, plot_lattice_folder, plot_data_folder


if __name__ == '__main__':
    app_field = np.zeros((N, NUM_STEPS))
    main(plot_period=10, app_field=app_field)
