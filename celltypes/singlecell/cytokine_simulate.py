import numpy as np

from cytokine_settings import build_model, DEFAULT_MODEL, APP_FIELD_STRENGTH, RUNS_SUBDIR_CYTOKINES
from singlecell_class import Cell
from singlecell_constants import NUM_STEPS, BETA
from singlecell_data_io import run_subdir_setup



def cytokine_sim(model_name=DEFAULT_MODEL, iterations=NUM_STEPS, applied_field_strength=APP_FIELD_STRENGTH , flag_write=False):

    # setup model and init cell class
    spin_labels, intxn_matrix, applied_field_const, init_state = build_model(model_name=model_name)
    cell = Cell(init_state, "model_%s" % model_name, memories_list=[], gene_list=spin_labels)

    # io
    if flag_write:
        current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = run_subdir_setup(run_subfolder=RUNS_SUBDIR_CYTOKINES)
        dirs = [current_run_folder, data_folder, plot_lattice_folder, plot_data_folder]
    else:
        dirs = None

    # simulate
    for step in xrange(iterations-1):
        print "cell steps:", cell.steps

        # plotting
        #if singlecell.steps % plot_period == 0:
        #    fig, ax, proj = singlecell.plot_projection(use_radar=True, pltdir=plot_lattice_folder)

        cell.update_state(intxn_matrix=intxn_matrix, app_field=applied_field_const, app_field_strength=applied_field_strength, randomize=False)

    # Write
    print "Writing state to file.."
    print cell.get_current_state()
    if flag_write:
        cell.write_state(data_folder)

    print "Done"
    return cell.get_state_array(), dirs


if __name__ == '__main__':
    cytokine_sim(iterations=10, flag_write=False)
