import numpy as np

from singlecell_class import Cell
from singlecell_constants import NUM_FULL_STEPS, BURST_ERROR_PERIOD, APP_FIELD_STRENGTH, BETA, ASYNC_BATCH
from singlecell_data_io import run_subdir_setup, runinfo_append
from singlecell_fields import field_setup
from singlecell_simsetup import singlecell_simsetup, unpack_simsetup

"""
NOTES:
- projection method seems to be behaving correctly
- TODO: test vs Fig 1E Mehta 2014
- in hopfield sim, at normal temps it jumps immediately to much more stable state and stays there
"""


def singlecell_sim(init_state=None, init_id=None, iterations=NUM_FULL_STEPS, beta=BETA, simsetup=None,
                   field_protocol=None, field_level=None, flag_burst_error=False, flag_write=True,
                   analysis_subdir=None, plot_period=10, verbose=True):
    """
    init_state: N x 1
    init_id: None, or memory label like 'esc', or arbitrary label (e.g. 'All on')
    iterations: main simulation loop duration
    field_protocol: label for call field_setup to build field dict for applied field
    flag_burst_error: if True, randomly flip some TFs at each BURST_ERROR_PERIOD (see ...constants.py)
    flag_write: False only if want to avoid saving state to file
    analysis_subdir: use to store data for non-standard runs
    plot_period: period at which to plot cell state projection onto memory subspace
    """
    # TODO: if dirs is None then do run subdir setup (just current run dir?)
    # IO setup
    if flag_write:
        io_dict = run_subdir_setup(run_subfolder=analysis_subdir)
    else:
        if verbose:
            print "Warning: flag_write set to False -- nothing will be saved"
        io_dict = None

    # simsetup unpack
    if simsetup is None:
        simsetup = singlecell_simsetup()
    N, P, gene_labels, memory_labels, gene_id, celltype_id, xi, _, a_inv, intxn_matrix, _ = unpack_simsetup(simsetup)

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
    field_dict = field_setup(simsetup, protocol=field_protocol, level=field_level)
    assert not field_dict['time_varying']  # TODO not yet supported
    app_field = field_dict['app_field']
    app_field_strength = field_dict['app_field_strength']

    # Simulate
    for step in xrange(iterations-1):
        if verbose:
            print "cell steps:", singlecell.steps, " H(state) =", singlecell.get_energy(intxn_matrix=intxn_matrix)  # TODO need general intxn_matrix parent class
        # apply burst errors
        if flag_burst_error and step % BURST_ERROR_PERIOD == 0:
            singlecell.apply_burst_errors()
        # prep applied field TODO see if better speed to pass array of zeros and ditch all these if not None checks...
        if flag_write:
            if singlecell.steps % plot_period == 0:
                fig, ax, proj = singlecell.plot_projection(a_inv, xi, use_radar=True, pltdir=io_dict['plotdatadir'])
                fig, ax, proj = singlecell.plot_overlap(xi, use_radar=True, pltdir=io_dict['plotdatadir'])
        singlecell.update_state(intxn_matrix, beta=beta, app_field=app_field, app_field_strength=app_field_strength, async_batch=ASYNC_BATCH)

    # Write
    if verbose:
        print singlecell.get_current_state()
    if flag_write:
        if verbose:
            print "Writing state to file.."
        singlecell.write_state(io_dict['datadir'])
    if verbose:
        print io_dict['basedir']
        print "Done"
    return singlecell.get_state_array(), io_dict


if __name__ == '__main__':
    flag_write = True
    simsetup = singlecell_simsetup()
    singlecell_sim(init_id='Macrophage (A)', field_protocol='miR_21', field_level='level_3', plot_period=10,
                   iterations=50, simsetup=simsetup, flag_write=flag_write, beta=22.2)
