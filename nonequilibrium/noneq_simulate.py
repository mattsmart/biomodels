import numpy as np

from noneq_constants import NUM_STEPS, DEFAULT_N
from noneq_data_io import run_subdir_setup
from noneq_state import State


def state_simulate(init_state=None, init_id=None, N=DEFAULT_N, iterations=NUM_STEPS, intxn_matrix=None, app_field=None,
                   flag_makedir=True, flag_write=True, analysis_subdir=None, plot_period=10):

    """
    init_state: N x 1
    init_id: None, or memory label like 'esc', or arbitrary label (e.g. 'All_on')
    iterations: main simulation loop duration
    intxn_matrix: J_ij interaction elements for spin i influence on spin j
    app_field: size N x timesteps applied field array; column k is field to apply at timestep k
    flag_write: False only if want to avoid saving state to file
    analysis_subdir: use to store data for non-standard runs
    plot_period: period at which to plot cell state projection onto memory subspace
    """
    # TODO: if dirs is None then do run subdir setup (just current run dir?)
    # IO setup
    if flag_makedir:
        if analysis_subdir is None:
            current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup()
        else:
            current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup(run_subfolder=analysis_subdir)
    else:
        current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = [None]*4
        assert not flag_write

    # State setup
    if init_state is None:
        if init_id is None:
            init_id = "init_random"
            init_state = np.random.choice([-1, 1], size=(N,), p=[1./2, 1./2])
        else:
            init_state = 1 + np.zeros(N)  # start with all genes on
    assert(len(init_state)) == N
    state = State(init_state, init_id)

    # Interaction matrix setup
    if intxn_matrix is None:
        intxn_matrix = np.random.rand(N,N)

    # Input checks
    if app_field is not None:
        assert len(app_field) == N
        assert len(app_field[0]) == iterations
    else:
        app_field_timestep = None

    # Simulate
    for step in xrange(iterations-1):
        #print "cell steps:", state.steps
        #print "H(state) =", state.get_energy()

        # prep applied field TODO see if better speed to pass array of zeros and ditch all these if not None checks...
        if app_field is not None:
            app_field_timestep = app_field[:, step]

        # plotting
        """
        if state.steps % plot_period == 0:
            fig, ax, proj = singlecell.plot_projection(use_radar=True, pltdir=plot_lattice_folder)
        """
        #state.update_state(intxn_matrix, app_field=app_field_timestep)
        state.update_state(intxn_matrix, app_field=app_field_timestep, randomize=False)

    # Write
    if flag_write:
        print "Writing state to file.."
        print state.get_current_state()
        state.write_state(data_folder)
        print "Done"
    return state.get_state_array(), current_run_folder, data_folder, plot_lattice_folder, plot_data_folder


if __name__ == '__main__':
    #app_field = np.zeros((N, NUM_STEPS))
    state_simulate(plot_period=10, iterations=3)
