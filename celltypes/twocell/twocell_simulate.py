import singlecell.init_multiprocessing  # BEFORE numpy

import numpy as np
import os
import random
import matplotlib.pyplot as plt

from twocell_visualize import simple_vis, lattice_timeseries_proj_grid, lattice_timeseries_overlap, \
    lattice_timeseries_energy, lattice_timeseries_state_grid
from multicell.multicell_class import SpatialCell
from multicell.multicell_constants import VALID_EXOSOME_STRINGS
from singlecell.singlecell_constants import MEMS_MEHTA, MEMS_UNFOLD, BETA
from singlecell.singlecell_data_io import run_subdir_setup, runinfo_append
from singlecell.singlecell_fields import construct_app_field_from_genes
from singlecell.singlecell_functions import single_memory_projection_timeseries, hamiltonian, sorted_energies, label_to_state
from singlecell.singlecell_simsetup import singlecell_simsetup # N, P, XI, CELLTYPE_ID, CELLTYPE_LABELS, GENE_ID
from singlecell.singlecell_visualize import plot_state_prob_map, hypercube_visualize

EXOSOME_STRING = 'no_exo_field'
EXOSOME_PRUNE = 0.0
PLOT_PERIOD = 10

# TODO file IO
# TODO new full timeseries visualizations
# TODO compute indiv energies, interaction term, display in plot somehow neatly?


def twocell_sim_fast(lattice, simsetup, num_steps, beta=BETA, exostring=EXOSOME_STRING, exoprune=EXOSOME_PRUNE,
                     gamma=1.0, app_field=None, app_field_strength=0.0):

    cell_A = lattice[0][0]
    cell_B = lattice[0][1]
    # local fields initialization
    neighbours_A = [[0, 1]]
    neighbours_B = [[0, 0]]
    # initial condition vis
    for step in xrange(num_steps-1):
        # TODO could compare against whole model random update sequence instead of this block version
        app_field_step = app_field  # TODO housekeeping applied field; N vs N+M
        # update cell A
        total_field_A, _ = cell_A.get_local_exosome_field(lattice, None, None, exosome_string=exostring,
                                                          ratio_to_remove=exoprune, neighbours=neighbours_A)
        if simsetup['FIELD_SEND'] is not None:
            total_field_A += cell_A.get_local_paracrine_field(lattice, neighbours_A, simsetup)
        cell_A.update_state(simsetup['J'], beta=beta,
                            ext_field=total_field_A,
                            ext_field_strength=gamma,
                            app_field=app_field_step,
                            app_field_strength=app_field_strength)
        # update cell B
        total_field_B, _ = cell_B.get_local_exosome_field(lattice, None, None, exosome_string=exostring,
                                                          ratio_to_remove=exoprune, neighbours=neighbours_B)
        if simsetup['FIELD_SEND'] is not None:
            total_field_B += cell_B.get_local_paracrine_field(lattice, neighbours_B, simsetup)
        cell_B.update_state(simsetup['J'], beta=beta,
                            ext_field=total_field_B,
                            ext_field_strength=gamma,
                            app_field=app_field_step,
                            app_field_strength=app_field_strength)
    return lattice


def twocell_sim(lattice, simsetup, num_steps, data_dict, io_dict, beta=BETA, exostring=EXOSOME_STRING,
                exoprune=EXOSOME_PRUNE, gamma=1.0, app_field=None, app_field_strength=0.0, ioflag=True):

    cell_A = lattice[0][0]
    cell_B = lattice[0][1]
    # local fields initialization
    neighbours_A = [[0, 1]]
    neighbours_B = [[0, 0]]
    # initial condition vis
    if ioflag:
        simple_vis(lattice, simsetup, io_dict['plotlatticedir'], 'Initial condition', savemod='_%d' % 0)
    for step in xrange(num_steps-1):
        # TODO could compare against whole model random update sequence instead of this block version

        app_field_step = app_field  # TODO housekeeping applied field; N vs N+M

        # update cell A
        total_field_A, _ = cell_A.get_local_exosome_field(lattice, None, None, exosome_string=exostring,
                                                          ratio_to_remove=exoprune, neighbours=neighbours_A)
        if simsetup['FIELD_SEND'] is not None:
            total_field_A += cell_A.get_local_paracrine_field(lattice, neighbours_A, simsetup)
        cell_A.update_state(simsetup['J'], beta=beta,
                            ext_field=total_field_A,
                            ext_field_strength=gamma,
                            app_field=app_field_step,
                            app_field_strength=app_field_strength)
        if ioflag and num_steps % PLOT_PERIOD == 0:
            simple_vis(lattice, simsetup, io_dict['plotlatticedir'], 'Step %dA' % step, savemod='_%dA' % step)
        # update cell B
        total_field_B, _ = cell_B.get_local_exosome_field(lattice, None, None, exosome_string=exostring,
                                                          ratio_to_remove=exoprune, neighbours=neighbours_B)
        if simsetup['FIELD_SEND'] is not None:
            total_field_B += cell_B.get_local_paracrine_field(lattice, neighbours_B, simsetup)
        cell_B.update_state(simsetup['J'], beta=beta,
                            ext_field=total_field_B,
                            ext_field_strength=gamma,
                            app_field=app_field_step,
                            app_field_strength=app_field_strength)
        if ioflag and num_steps % PLOT_PERIOD == 0:
            simple_vis(lattice, simsetup, io_dict['plotlatticedir'], 'Step %dB' % step, savemod='_%dB' % step)

    # fill in data
    print 'simulation done; gathering data'
    if 'memory_proj_arr' in data_dict.keys():
        for memory_idx in range(simsetup['P']):
            data_dict['memory_proj_arr'][memory_idx][0, :] = \
                single_memory_projection_timeseries(cell_A.get_state_array(), memory_idx, simsetup['ETA'])
            data_dict['memory_proj_arr'][memory_idx][1, :] = \
                single_memory_projection_timeseries(cell_B.get_state_array(), memory_idx, simsetup['ETA'])
    if 'overlap' in data_dict.keys():
        data_dict['overlap'] = np.array([
            np.dot(cell_A.state_array[:, i], cell_B.state_array[:, i]) for i in xrange(num_steps)]) / simsetup['N']
    if 'grid_state_int' in data_dict.keys():
        # TODO
        print 'TODO grid_state_int data fill in'
        #data_dict['grid_state_int'] = np.zeros((2, num_steps), dtype=int)
    if 'multi_hamiltonian' in data_dict.keys():
        data_dict['single_hamiltonians'][0, :] = [hamiltonian(cell_A.state_array[:, i], simsetup['J'], field=app_field, fs=app_field_strength) for i in xrange(num_steps)]
        data_dict['single_hamiltonians'][1, :] = [hamiltonian(cell_B.state_array[:, i], simsetup['J'], field=app_field, fs=app_field_strength) for i in xrange(num_steps)]
        if simsetup['FIELD_SEND'] is not None:
            # TODO check the algebra here...
            W = simsetup['FIELD_SEND']
            WdotOne = np.dot(W, np.ones(simsetup['N']))
            WSym2 = (W + W.T)
            data_dict['unweighted_coupling_term'][:] = \
                [- 0.5 * np.dot( cell_A.state_array[:, i], np.dot(WSym2, cell_B.state_array[:, i]) )
                 - 0.5 * np.dot(WdotOne, cell_A.state_array[:, i] + cell_B.state_array[:, i])
                 for i in xrange(num_steps)]
        data_dict['multi_hamiltonian'] = (data_dict['single_hamiltonians'][0, :] + data_dict['single_hamiltonians'][1, :]) + gamma * data_dict['unweighted_coupling_term']
    return lattice, data_dict, io_dict


def twocell_simprep(simsetup, num_steps, beta=BETA, exostring=EXOSOME_STRING, exoprune=EXOSOME_PRUNE, gamma=1.0,
                    app_field=None, app_field_strength=0.0):
    """
    Prep lattice (of two cells), fields, and IO
    """
    # check args
    assert type(num_steps) is int
    assert exostring in VALID_EXOSOME_STRINGS
    assert 0.0 <= exoprune < 1.0
    assert 0.0 <= gamma < 10.0

    cell_a_init = simsetup['XI'][:, 0]
    cell_b_init = simsetup['XI'][:, 0]
    lattice = [[SpatialCell(cell_a_init, 'Cell A', [0, 0], simsetup),
                SpatialCell(cell_b_init, 'Cell B', [0, 1], simsetup)]]  # list of list to conform to multicell slattice funtions

    # app fields initialization
    app_field_step = None  # TODO housekeeping applied field; N vs N+M

    # setup io dict
    io_dict = run_subdir_setup()
    info_list = [['memories_path', simsetup['memories_path']], ['script', 'twocell_simulate.py'],
                 ['num_steps', num_steps], ['fieldstring', exostring], ['field_remove_ratio', exoprune],
                 ['app_field_strength', app_field_strength], ['ext_field_strength', gamma], ['app_field', app_field],
                 ['beta', beta], ['random_mem', simsetup['random_mem']], ['random_W', simsetup['random_W']]]
    runinfo_append(io_dict, info_list, multi=True)
    # conditionally store random mem and W
    np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_XI.txt', simsetup['XI'], delimiter=',', fmt='%d')
    if simsetup['FIELD_SEND'] is not None:
        np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_W.txt', simsetup['FIELD_SEND'], delimiter=',', fmt='%.4f')
    else:
        runinfo_append(io_dict, [simsetup['FIELD_SEND'], None], multi=False)

    # setup data dictionary
    # TODO increase timstep resolution by x2 -- every cell update (see sequence info...)
    data_dict = {}
    store_state_int = True
    store_memory_proj_arr = True
    store_overlap = True
    store_energy = True
    # TODO what data are we storing?
    # store projection onto each memory
    if store_memory_proj_arr:
        data_dict['memory_proj_arr'] = {}
        for idx in range(simsetup['P']):
            data_dict['memory_proj_arr'][idx] = np.zeros((2, num_steps))
    # store cell-cell overlap as scalar
    if store_overlap:
        data_dict['overlap'] = np.zeros(num_steps)
    # store state as int (compressed)
    if store_state_int:
        assert simsetup['N'] < 12
        data_dict['grid_state_int'] = np.zeros((2, num_steps), dtype=int)
    if store_energy:
        data_dict['single_hamiltonians'] = np.zeros((2, num_steps))
        data_dict['multi_hamiltonian'] = np.zeros(num_steps)
        data_dict['unweighted_coupling_term'] = np.zeros(num_steps)

    # run the simulation
    lattice, data_dict, io_dict = \
        twocell_sim(lattice, simsetup, num_steps, data_dict, io_dict, beta=beta, exostring=exostring, exoprune=exoprune,
                    gamma=gamma, app_field=app_field, app_field_strength=app_field_strength)

    # check the data dict
    """
    for data_idx, memory_idx in enumerate(data_dict['memory_proj_arr'].keys()):
        print data_dict['memory_proj_arr'][memory_idx]
        plt.plot(data_dict['memory_proj_arr'][memory_idx].T)
        plt.ylabel('Projection of all cells onto type: %s' % simsetup['CELLTYPE_LABELS'][memory_idx])
        plt.xlabel('Time (full lattice steps)')
        plt.savefig(io_dict['plotdatadir'] + os.sep + '%s_%s_n%d_t%d_proj%d_remove%.2f_exo%.2f.png' %
                    (exosome_string, buildstring, gridsize, num_steps, memory_idx, field_remove_ratio, ext_field_strength))
        plt.clf()  #plt.show()
    """
    # additional visualizations
    lattice_timeseries_state_grid(lattice, simsetup, io_dict['plotdatadir'], savemod='')
    lattice_timeseries_proj_grid(data_dict, simsetup, io_dict['plotdatadir'], savemod='')
    lattice_timeseries_proj_grid(data_dict, simsetup, io_dict['plotdatadir'], savemod='conj', conj=True)
    lattice_timeseries_overlap(data_dict, simsetup, io_dict['plotdatadir'], savemod='')
    # TODO include app field energy
    lattice_timeseries_energy(data_dict, simsetup, io_dict['plotdatadir'], savemod='')
    return lattice, data_dict, io_dict


if __name__ == '__main__':
    HOUSEKEEPING = 0
    KAPPA = 100

    random_mem = False
    random_W = False
    #simsetup = singlecell_simsetup(unfolding=False, random_mem=random_mem, random_W=random_W, npzpath=MEMS_MEHTA)
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, npzpath=MEMS_UNFOLD,
                                   housekeeping=HOUSEKEEPING)
    print 'note: N =', simsetup['N']
    steps = 20
    beta = 2.0  # 2.0

    exostring = "no_exo_field"  # on/off/all/no_exo_field, note e.g. 'off' means send info about 'off' genes only
    exoprune = 0.0              # amount of exosome field idx to randomly prune from each cell
    gamma = 0.0                 # global EXT_FIELD_STRENGTH tunes exosomes AND sent field

    app_field = None
    if KAPPA > 0 and HOUSEKEEPING > 0:
        app_field = np.zeros(simsetup['N'])
        app_field[-HOUSEKEEPING:] = 1.0

    lattice, data_dict, io_dict = \
        twocell_simprep(simsetup, steps, beta=beta, exostring=exostring, exoprune=exoprune, gamma=gamma,
                        app_field=app_field, app_field_strength=KAPPA)
