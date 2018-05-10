import numpy as np

from cytokine_settings import build_intracell_model, DEFAULT_CYTOKINE_MODEL, APP_FIELD_STRENGTH, RUNS_SUBDIR_CYTOKINES
from singlecell_class import Cell
from singlecell_constants import NUM_STEPS, BETA
from singlecell_data_io import run_subdir_setup


def cytokine_sim(model_name=DEFAULT_CYTOKINE_MODEL, iterations=NUM_STEPS, beta=BETA,
                 applied_field_strength=APP_FIELD_STRENGTH, init_state_force=None, flag_write=False, flag_print=False):

    # setup model and init cell class
    spin_labels, intxn_matrix, applied_field_const, init_state = build_intracell_model(model_name=model_name)
    if init_state_force is None:
        cell = Cell(init_state, "model_%s" % model_name, memories_list=[], gene_list=spin_labels)
    else:
        cell = Cell(init_state_force, "model_%s_init_state_forced" % model_name, memories_list=[], gene_list=spin_labels)

    # io
    if flag_write:
        current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup(run_subfolder=RUNS_SUBDIR_CYTOKINES)
        dirs = [current_run_folder, data_folder, plot_lattice_folder, plot_data_folder]
    else:
        dirs = None

    # simulate
    for step in xrange(iterations-1):

        if flag_print:
            print cell.steps, "cell steps:", cell.get_current_state()

        # plotting
        #if singlecell.steps % plot_period == 0:
        #    fig, ax, proj = singlecell.plot_projection(use_radar=True, pltdir=plot_lattice_folder)

        cell.update_state(intxn_matrix=intxn_matrix, beta=beta, app_field=applied_field_const, app_field_strength=applied_field_strength, randomize=False)

    # end state
    if flag_print:
        print cell.steps + 1, "cell steps:", cell.get_current_state()

    # Write
    if flag_write:
        print "Writing state to file.."
        cell.write_state(data_folder)

    if flag_print:
        print "Done"
    return cell.get_state_array(), dirs


if __name__ == '__main__':
    cytokine_sim(iterations=20, applied_field_strength=0.0, flag_write=False, flag_print=True)
