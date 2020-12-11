import numpy as np
import os
import random
import matplotlib.pyplot as plt

from multicell_constants import GRIDSIZE, SEARCH_RADIUS_CELL, NUM_LATTICE_STEPS, VALID_BUILDSTRINGS, MEANFIELD, \
    VALID_EXOSOME_STRINGS, EXOSTRING, BUILDSTRING, LATTICE_PLOT_PERIOD, FIELD_REMOVE_RATIO
from multicell_lattice import build_lattice_main, get_cell_locations, prep_lattice_data_dict, write_state_all_cells, \
    write_grid_state_int, write_general_arr, read_general_arr
from multicell_metrics import calc_lattice_energy, calc_compression_ratio, get_state_of_lattice
from multicell_visualize import lattice_uniplotter, reference_overlap_plotter, lattice_projection_composite
from singlecell.singlecell_constants import EXT_FIELD_STRENGTH, APP_FIELD_STRENGTH, BETA
from singlecell.singlecell_data_io import run_subdir_setup, runinfo_append
from singlecell.singlecell_fields import construct_app_field_from_genes
from singlecell.singlecell_simsetup import singlecell_simsetup # N, P, XI, CELLTYPE_ID, CELLTYPE_LABELS, GENE_ID


def run_sim(lattice, num_lattice_steps, data_dict, io_dict, simsetup, exosome_string=EXOSTRING, field_remove_ratio=0.0,
            ext_field_strength=EXT_FIELD_STRENGTH, app_field=None, app_field_strength=APP_FIELD_STRENGTH, beta=BETA,
            plot_period=LATTICE_PLOT_PERIOD, flag_uniplots=False, state_int=False, meanfield=MEANFIELD):
    """
    Form of data_dict:
        {'memory_proj_arr':
            {memory_idx: np array [N x num_steps] of projection each grid cell onto memory idx}
         'grid_state_int': n x n x num_steps of int at each site (int is inverse of binary string from state)}
    Notes:
        -can replace update_with_signal_field with update_state to simulate ensemble of non-intxn n**2 cells
    """

    # Input checks
    n = len(lattice)
    assert n == len(lattice[0])  # work with square lattice for simplicity
    num_cells = n * n
    assert SEARCH_RADIUS_CELL < n / 2.0  # to prevent signal double counting

    if app_field is not None:
        if len(app_field.shape) > 1:
            assert app_field.shape[0] == simsetup['N']
            assert len(app_field[1]) == num_lattice_steps
        else:
            app_field = np.array([app_field for _ in range(num_lattice_steps)]).T
        app_field_step = app_field[:, 0]
    else:
        app_field_step = None

    cell_locations = get_cell_locations(lattice, n)
    loc_to_idx = {pair: idx for idx, pair in enumerate(cell_locations)}
    memory_idx_list = list(data_dict['memory_proj_arr'].keys())

    # plot initial state of the lattice
    if flag_uniplots:
        for mem_idx in memory_idx_list:
            lattice_uniplotter(lattice, 0, n, io_dict['latticedir'], mem_idx, simsetup)
    # get data for initial state of the lattice
    for loc in cell_locations:
        cell = lattice[loc[0]][loc[1]]
        cell.get_memories_projection(simsetup['A_INV'], simsetup['XI'])
        data_dict['grid_state_int'][loc[0], loc[1], 0] = cell.get_current_label()
        for mem_idx in memory_idx_list:
            proj = cell.get_memories_projection(simsetup['A_INV'], simsetup['XI'])
            data_dict['memory_proj_arr'][mem_idx][loc_to_idx[loc], 0] = proj[mem_idx]
    data_dict['lattice_energy'][0, :] = calc_lattice_energy(lattice, simsetup, app_field_step, app_field_strength, ext_field_strength,
                                                            SEARCH_RADIUS_CELL, field_remove_ratio, exosome_string, meanfield)
    data_dict['compressibility_full'][0, :] = calc_compression_ratio(get_state_of_lattice(lattice, simsetup, datatype='full'),
                                                                     eta_0=None, datatype='full', elemtype=np.int, method='manual')

    # initial condition plot
    lattice_projection_composite(lattice, 0, n, io_dict['latticedir'], simsetup, state_int=state_int)
    reference_overlap_plotter(lattice, 0, n, io_dict['latticedir'], simsetup, state_int=state_int)
    if flag_uniplots:
        for mem_idx in memory_idx_list:
            lattice_uniplotter(lattice, 0, n, io_dict['latticedir'], mem_idx, simsetup)

    # special update method for meanfield case (infinite search radius)
    if meanfield:
        assert exosome_string == 'no_exo_field'  # TODO careful: not clear best way to update exo field as cell state changes in a time step, refactor exo fn?
        print('Initializing mean field...')
        # TODO decide if want scale factor to be rescaled by total popsize (i.e. *mean*field or total field?)
        state_total = np.zeros(simsetup['N'])
        field_global = np.zeros(simsetup['N'])
        neighbours = [[a, b] for a in range(len(lattice[0])) for b in range(len(lattice))]  # TODO ok that cell is neighbour with self as well? remove diag
        if simsetup['FIELD_SEND'] is not None:
            for loc in neighbours:
                state_total += lattice[loc[0]][loc[1]].get_current_state()
            state_total_01 = (state_total + num_cells) / 2
            field_paracrine = np.dot(simsetup['FIELD_SEND'], state_total_01)
            field_global += field_paracrine
        if exosome_string != 'no_exo_field':
            field_exo, _ = lattice[0][0].get_local_exosome_field(lattice, None, None, exosome_string=exosome_string,
                                                                 ratio_to_remove=field_remove_ratio,
                                                                 neighbours=neighbours)
            field_global += field_exo

    for turn in range(1, num_lattice_steps):
        print('Turn ', turn)
        random.shuffle(cell_locations)
        for idx, loc in enumerate(cell_locations):
            cell = lattice[loc[0]][loc[1]]
            if app_field is not None:
                app_field_step = app_field[:, turn]
            if meanfield:
                cellstate_pre = np.copy(cell.get_current_state())
                cell.update_with_meanfield(simsetup['J'], field_global, beta=beta, app_field=app_field_step,
                                           ext_field_strength=ext_field_strength, app_field_strength=app_field_strength)
                state_total += (cell.get_current_state() - cellstate_pre)  # TODO update field_avg based on new state TODO test
                state_total_01 = (state_total + num_cells) / 2
                field_global = np.dot(simsetup['FIELD_SEND'], state_total_01)
                print(field_global)
                print(state_total)
            else:
                cell.update_with_signal_field(lattice, SEARCH_RADIUS_CELL, n, simsetup['J'], simsetup, beta=beta,
                                              exosome_string=exosome_string, ratio_to_remove=field_remove_ratio,
                                              ext_field_strength=ext_field_strength, app_field=app_field_step,
                                              app_field_strength=app_field_strength)
            if state_int:
                data_dict['grid_state_int'][loc[0], loc[1], turn] = cell.get_current_label()
            proj = cell.get_memories_projection(simsetup['A_INV'], simsetup['XI'])
            for mem_idx in memory_idx_list:
                data_dict['memory_proj_arr'][mem_idx][loc_to_idx[loc], turn] = proj[mem_idx]
            if turn % (120*plot_period) == 0:  # plot proj visualization of each cell (takes a while; every k lat plots)
                fig, ax, proj = cell.plot_projection(simsetup['A_INV'], simsetup['XI'], use_radar=False, pltdir=io_dict['latticedir'])

        if turn % plot_period == 0:  # plot the lattice
            lattice_projection_composite(lattice, turn, n, io_dict['latticedir'], simsetup, state_int=state_int)
            reference_overlap_plotter(lattice, turn, n, io_dict['latticedir'], simsetup, state_int=state_int)
            #if flag_uniplots:
            #    for mem_idx in memory_idx_list:
            #        lattice_uniplotter(lattice, turn, n, io_dict['latticedir'], mem_idx, simsetup)

        # compute lattice properties
        # TODO 1 - consider lattice energy at each cell update (not lattice update)
        # TODO 2 - speedup lattice energy calc by using info from state update calls...
        data_dict['lattice_energy'][turn, :] = calc_lattice_energy(
            lattice, simsetup, app_field_step, app_field_strength, ext_field_strength, SEARCH_RADIUS_CELL,
            field_remove_ratio, exosome_string, meanfield)
        data_dict['compressibility_full'][turn, :] = calc_compression_ratio(
            get_state_of_lattice(lattice, simsetup, datatype='full'),
            eta_0=data_dict['compressibility_full'][0,2], datatype='full', elemtype=np.int, method='manual')

    return lattice, data_dict, io_dict


def mc_sim(simsetup, gridsize=GRIDSIZE, num_steps=NUM_LATTICE_STEPS, buildstring=BUILDSTRING, exosome_string=EXOSTRING,
           field_remove_ratio=FIELD_REMOVE_RATIO, ext_field_strength=EXT_FIELD_STRENGTH, app_field=None,
           app_field_strength=APP_FIELD_STRENGTH, beta=BETA, plot_period=LATTICE_PLOT_PERIOD, state_int=False,
           meanfield=MEANFIELD):

    # check args
    assert type(gridsize) is int
    assert type(num_steps) is int
    assert type(plot_period) is int
    assert buildstring in VALID_BUILDSTRINGS
    assert exosome_string in VALID_EXOSOME_STRINGS
    assert 0.0 <= field_remove_ratio < 1.0
    assert 0.0 <= ext_field_strength < 10.0

    # setup io dict
    io_dict = run_subdir_setup()
    if meanfield:
        search_radius_txt = 'None'
    else:
        search_radius_txt = SEARCH_RADIUS_CELL
    info_list = [['memories_path', simsetup['memories_path']], ['script', 'multicell_simulate.py'], ['gridsize', gridsize],
                 ['num_steps', num_steps], ['buildstring', buildstring], ['fieldstring', exosome_string],
                 ['field_remove_ratio', field_remove_ratio], ['app_field_strength', app_field_strength],
                 ['ext_field_strength', ext_field_strength], ['app_field', app_field], ['beta', beta],
                 ['search_radius', search_radius_txt], ['random_mem', simsetup['random_mem']],
                 ['random_W', simsetup['random_W']], ['meanfield', meanfield], ['housekeeping', simsetup['K']]]
    runinfo_append(io_dict, info_list, multi=True)
    # conditionally store random mem and W
    np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_XI.txt', simsetup['XI'], delimiter=',', fmt='%d')
    np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_W.txt', simsetup['FIELD_SEND'], delimiter=',', fmt='%.4f')

    # setup lattice IC
    flag_uniplots = False
    if buildstring == "mono":
        type_1_idx = 0
        list_of_type_idx = [type_1_idx]
    if buildstring == "dual":
        type_1_idx = 0
        type_2_idx = 1
        list_of_type_idx = [type_1_idx, type_2_idx]
    if buildstring == "memory_sequence":
        flag_uniplots = False
        list_of_type_idx = list(range(simsetup['P']))
        #random.shuffle(list_of_type_idx)  # TODO shuffle or not?
    if buildstring == "random":
        flag_uniplots = False
        list_of_type_idx = list(range(simsetup['P']))
    lattice = build_lattice_main(gridsize, list_of_type_idx, buildstring, simsetup)
    #print list_of_type_idx

    # prep data dictionary
    data_dict = {}
    data_dict = prep_lattice_data_dict(gridsize, num_steps, list_of_type_idx, buildstring, data_dict)
    if state_int:
        data_dict['grid_state_int'] = np.zeros((gridsize, gridsize, num_steps), dtype=int)

    # run the simulation
    lattice, data_dict, io_dict = \
        run_sim(lattice, num_steps, data_dict, io_dict, simsetup, exosome_string=exosome_string, field_remove_ratio=field_remove_ratio,
                ext_field_strength=ext_field_strength, app_field=app_field, app_field_strength=app_field_strength,
                beta=beta, plot_period=plot_period, flag_uniplots=flag_uniplots, state_int=state_int, meanfield=meanfield)

    # check the data dict
    for data_idx, memory_idx in enumerate(data_dict['memory_proj_arr'].keys()):
        print(data_dict['memory_proj_arr'][memory_idx])
        plt.plot(data_dict['memory_proj_arr'][memory_idx].T)
        plt.ylabel('Projection of all cells onto type: %s' % simsetup['CELLTYPE_LABELS'][memory_idx])
        plt.xlabel('Time (full lattice steps)')
        plt.savefig(io_dict['plotdatadir'] + os.sep + '%s_%s_n%d_t%d_proj%d_remove%.2f_exo%.2f.png' %
                    (exosome_string, buildstring, gridsize, num_steps, memory_idx, field_remove_ratio, ext_field_strength))
        plt.clf()  #plt.show()

    # write and plot cell state timeseries
    # TODO convert to 'write data dict' and 'plot data dict' fn calls
    #write_state_all_cells(lattice, io_dict['datadir'])
    if state_int:
        write_grid_state_int(data_dict['grid_state_int'], io_dict['datadir'])
    if 'lattice_energy' in list(data_dict.keys()):
        write_general_arr(data_dict['lattice_energy'], io_dict['datadir'], 'lattice_energy', txt=True, compress=False)
        plt.plot(data_dict['lattice_energy'][:, 0], '--ok', label=r'$H_{\mathrm{total}}$')
        plt.plot(data_dict['lattice_energy'][:, 1], '--b', alpha=0.7, label=r'$H_{\mathrm{self}}$')
        plt.plot(data_dict['lattice_energy'][:, 2], '--g', alpha=0.7, label=r'$H_{\mathrm{app}}$')
        plt.plot(data_dict['lattice_energy'][:, 3], '--r', alpha=0.7, label=r'$H_{\mathrm{pairwise}}$')
        plt.plot(data_dict['lattice_energy'][:, 0] - data_dict['lattice_energy'][:, 2], '--o', color='gray',
                 label=r'$H_{\mathrm{total}} - H_{\mathrm{app}}$')
        plt.title(r'Multicell hamiltonian over time')
        plt.ylabel(r'Lattice energy')
        plt.xlabel(r'$t$ (lattice steps)')
        plt.legend()
        plt.savefig(io_dict['plotdatadir'] + os.sep + '%s_%s_n%d_t%d_hamiltonian_remove%.2f_exo%.2f.png' %
                    (exosome_string, buildstring, gridsize, num_steps, field_remove_ratio, ext_field_strength))
        # zoom on relevant part
        ylow = min(np.min(data_dict['lattice_energy'][:, [1,3]]),
                   np.min(data_dict['lattice_energy'][:, 0] - data_dict['lattice_energy'][:, 2]))
        yhigh = max(np.max(data_dict['lattice_energy'][:, [1,3]]),
                    np.max(data_dict['lattice_energy'][:, 0] - data_dict['lattice_energy'][:, 2]))
        plt.ylim(ylow - 0.1, yhigh + 0.1)
        plt.savefig(io_dict['plotdatadir'] + os.sep + '%s_%s_n%d_t%d_hamiltonianZoom_remove%.2f_exo%.2f.png' %
                    (exosome_string, buildstring, gridsize, num_steps, field_remove_ratio, ext_field_strength))
        plt.clf()  # plt.show()
    if 'compressibility_full' in list(data_dict.keys()):
        write_general_arr(data_dict['compressibility_full'], io_dict['datadir'], 'compressibility_full', txt=True, compress=False)
        plt.plot(data_dict['compressibility_full'][:,0], '--o', color='orange')
        plt.title(r'File compressibility ratio of the full lattice spin state')
        plt.ylabel(r'$\eta(t)/\eta_0$')
        plt.axhline(y=1.0, ls='--', color='k')

        ref_0 = calc_compression_ratio(x=np.zeros((len(lattice), len(lattice[0]), simsetup['N']), dtype=int),
                                       eta_0=data_dict['compressibility_full'][0,2], datatype='full', elemtype=np.int, method='manual')
        ref_1 = calc_compression_ratio(x=np.ones((len(lattice), len(lattice[0]), simsetup['N']), dtype=int),
                                       eta_0=data_dict['compressibility_full'][0,2], datatype='full', elemtype=np.int, method='manual')
        plt.axhline(y=ref_0[0], ls='-.', color='gray')
        plt.axhline(y=ref_1[0], ls='-.', color='blue')
        print(ref_0,ref_0,ref_0,ref_0, 'is', ref_0, 'vs', ref_1)
        plt.xlabel(r'$t$ (lattice steps)')
        plt.ylim(-0.05, 1.01)
        plt.savefig(io_dict['plotdatadir'] + os.sep + '%s_%s_n%d_t%d_comp_remove%.2f_exo%.2f.png' %
                    (exosome_string, buildstring, gridsize, num_steps, field_remove_ratio, ext_field_strength))
        plt.clf()  # plt.show()

    print("\nMulticell simulation complete - output in %s" % io_dict['basedir'])
    return lattice, data_dict, io_dict


if __name__ == '__main__':
    random_mem = False
    random_W = False
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, curated=True, housekeeping=0)

    n = 20  # global GRIDSIZE
    steps = 10  # global NUM_LATTICE_STEPS
    buildstring = "mono"  # mono/dual/memory_sequence/random
    fieldstring = "no_exo_field"  # on/off/all/no_exo_field, note e.g. 'off' means send info about 'off' genes only
    meanfield = False  # set to true to use infinite signal distance (no neighbour searching; track mean field)
    fieldprune = 0.0  # amount of external field idx to randomly prune from each cell
    ext_field_strength = 0.1  #  / (n*n) * 8                                # global GAMMA = EXT_FIELD_STRENGTH tunes exosomes AND sent field
    #app_field = construct_app_field_from_genes(IPSC_EXTENDED_GENES_EFFECTS, simsetup['GENE_ID'], num_steps=steps)        # size N x timesteps or None

    app_field = None
    # housekeeping genes block
    KAPPA = 10.0
    if KAPPA > 0:
        # housekeeping auto (via model extension)
        app_field = np.zeros(simsetup['N'])
        if simsetup['K'] > 0:
            app_field[-simsetup['K']:] = 1.0
            print(app_field)
        else:
            print('Note gene 0 (on), 1 (on), 2 (on) are HK in A1 memories')
            print('Note gene 4 (off), 5 (on) are HK in C1 memories')
            app_field[4] = 1.0
            app_field[5] = 1.0

    plot_period = 1
    state_int = True
    beta = BETA  # 2.0
    mc_sim(simsetup, gridsize=n, num_steps=steps, buildstring=buildstring, exosome_string=fieldstring,
           field_remove_ratio=fieldprune, ext_field_strength=ext_field_strength, app_field=app_field,
           app_field_strength=KAPPA, beta=beta, plot_period=plot_period, state_int=state_int,
           meanfield=meanfield)
    """
    for beta in [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 4.0, 5.0, 10.0, 100.0]:
        mc_sim(simsetup, gridsize=n, num_steps=steps, buildstring=buildstring, exosome_string=fieldstring,
               field_remove_ratio=fieldprune, ext_field_strength=ext_field_strength, app_field=app_field,
               app_field_strength=app_field_strength, beta=beta, plot_period=plot_period, state_int=state_int, meanfield=meanfield)
    """
