import numpy as np

from singlecell_class import Cell
from singlecell_constants import NUM_STEPS, BURST_ERROR_PERIOD, APP_FIELD_STRENGTH, BETA
from singlecell_data_io import run_subdir_setup
from singlecell_simsetup import N, XI, J, CELLTYPE_ID, CELLTYPE_LABELS, GENE_LABELS

"""
NOTES:
- projection method seems to be behaving correctly
- TODO: test vs Fig 1E Mehta 2014
- in hopfield sim, at normal temps it jumps immediately to much more stable state and stays there
"""


def singlecell_sim(init_state=None, init_id=None, iterations=NUM_STEPS, beta=BETA, xi=XI, intxn_matrix=J,
                   celltype_id=CELLTYPE_ID, memory_labels=CELLTYPE_LABELS, gene_labels=GENE_LABELS,
                   app_field=None, app_field_strength=APP_FIELD_STRENGTH, flag_burst_error=False, flag_write=True,
                   analysis_subdir=None, plot_period=10, verbose=True):
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
    if flag_write:
        if analysis_subdir is None:
            current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup()
        else:
            current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup(run_subfolder=analysis_subdir)
    else:
        if verbose:
            print "Warning: flag_write set to False -- nothing will be saved"
        current_run_folder = None
        data_folder = None
        plot_lattice_folder = None
        plot_data_folder = None

    # Cell setup
    N = xi.shape[0]
    if init_state is None:
        if init_id is None:
            init_id = "All_on"
            init_state = 1 + np.zeros(N)  # start with all genes on
        else:
            init_state = xi[:, celltype_id[init_id]]
    singlecell = Cell(init_state, init_id, memories_list=memory_labels, gene_list=gene_labels)

    # Input checks
    if app_field is not None:
        assert len(app_field) == N
        assert len(app_field[0]) == iterations
    else:
        app_field_timestep = None

    # Simulate
    for step in xrange(iterations-1):
        if verbose:
            print "cell steps:", singlecell.steps, " H(state) =", singlecell.get_energy(intxn_matrix=intxn_matrix)  # TODO need general intxn_matrix parent class
        # apply burst errors
        if flag_burst_error and step % BURST_ERROR_PERIOD == 0:
            singlecell.apply_burst_errors()
        # prep applied field TODO see if better speed to pass array of zeros and ditch all these if not None checks...
        if app_field is not None:
            app_field_timestep = app_field[:, step]
        if flag_write:
            if singlecell.steps % plot_period == 0:
                fig, ax, proj = singlecell.plot_projection(use_radar=True, pltdir=plot_lattice_folder)
        singlecell.update_state(beta=beta, intxn_matrix=intxn_matrix, app_field=app_field_timestep,
                                app_field_strength=app_field_strength)

    # Write
    if verbose:
        print singlecell.get_current_state()
    if flag_write:
        if verbose:
            print "Writing state to file.."
        singlecell.write_state(data_folder)
    if verbose:
        print "Done"
    return singlecell.get_state_array(), current_run_folder, data_folder, plot_lattice_folder, plot_data_folder


if __name__ == '__main__':
    flag_write = False
    app_field = np.zeros((N, NUM_STEPS))
    singlecell_sim(plot_period=10, app_field=app_field, flag_write=flag_write)
