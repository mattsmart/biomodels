import numpy as np
import os
import pickle
import shutil

from multicell.graph_helper import state_load
from multicell.multicell_metrics import calc_graph_energy
from multicell.multicell_simulate import Multicell
from singlecell.singlecell_simsetup import singlecell_simsetup
from utils.file_io import RUNS_FOLDER, INPUT_FOLDER


def aggregate_manyruns(runs_basedir, agg_subdir='aggregate', fastplot=True):
    agg_dir = runs_basedir + os.sep + agg_subdir
    if not os.path.exists(agg_dir):
        os.mkdir(agg_dir)

    # Step 0) get all the run directories
    fpaths = [runs_basedir + os.sep + a for a in os.listdir(runs_basedir)]
    run_dirs = [a for a in fpaths
                if os.path.isdir(a) and os.path.basename(a) != 'aggregate']

    # Step 1) get info from the first run directory (require at least one)
    ppath = runs_basedir + os.sep + 'multicell_template.pkl'
    with open(ppath, 'rb') as pickle_file:
        multicell_template = pickle.load(pickle_file)  # Unpickling the object
    num_genes = multicell_template.num_genes
    num_cells = multicell_template.num_cells
    total_spins = num_genes * num_cells

    # Step 2) aggregate file containing all the fixed points
    # X_aggregate.npz -- 2D, total_spins x num_runs, full state of each FP
    # X_energies.npz  -- 2D,           5 x num_runs, energy tuple of each FP
    # if fastplot, produce plot of each state
    fixedpoints_ensemble = np.zeros((total_spins, num_runs), dtype=int)
    energies = np.zeros((5, num_runs), dtype=float)
    for i, run_dir in enumerate(run_dirs):
        print(i, run_dir)
        fpath = run_dir + os.sep + 'states' + os.sep + 'X_last.npz'
        # 2.1) get final state
        X = state_load(fpath, cells_as_cols=False, num_genes=num_genes,
                       num_cells=num_cells, txt=False)
        fixedpoints_ensemble[:, i] = X

        # 2.2) get state energy for bokeh
        step_hack = 0  # TODO care this will break if class has time-varying applied field
        multicell_template.graph_state_arr[:, step_hack] = X[:]
        assert np.array_equal(multicell_template.field_applied,
                              np.zeros((total_spins, multicell_template.total_steps)))
        state_energy = calc_graph_energy(multicell_template, step_hack, norm=True)
        print(state_energy)
        energies[:, i] = state_energy
        # 2.3) get state image for bokeh
        if fastplot:
            plot_state_simple(X)  # TODO

    np.savez_compressed(agg_dir + os.sep + 'X_aggregate', fixedpoints_ensemble)
    np.savez_compressed(agg_dir + os.sep + 'X_energy', energies)


def plot_state_simple(X):
    print('TODO plot')
    return


if __name__ == '__main__':

    multirun_name = 'multicell_manyruns'
    multirun_path = RUNS_FOLDER + os.sep + multirun_name
    aggregate_data = True
    #assert not os.path.exists(multirun_path)
    if os.path.exists(multirun_path):
        shutil.rmtree(multirun_path)

    # 1) create simsetup
    simsetup_seed = 0
    curated = False
    random_mem = False        # TODO incorporate seed in random XI
    random_W = False          # TODO incorporate seed in random W
    W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_maze.txt'
    simsetup_main = singlecell_simsetup(
        unfolding=True, random_mem=random_mem, random_W=random_W, curated=curated, housekeeping=0)
    if W_override_path is not None:
        print('Note: in main, overriding W from file...')
        explicit_W = np.loadtxt(W_override_path, delimiter=',')
        simsetup_main['FIELD_SEND'] = explicit_W

    print("simsetup checks:")
    print("\tsimsetup['N'],", simsetup_main['N'])
    print("\tsimsetup['P'],", simsetup_main['P'])

    # setup 2.1) multicell sim core parameters
    num_cells = 10**2          # global GRIDSIZE
    total_steps = 100           # global NUM_LATTICE_STEPS
    plot_period = 1
    flag_state_int = True
    flag_blockparallel = False
    if aggregate_data:
        assert not flag_blockparallel
    beta = 2000.0
    gamma = 0.0               # i.e. field_signal_strength
    kappa = 0.0                # i.e. field_applied_strength

    # setup 2.2) graph options
    autocrine = False
    graph_style = 'lattice_square'
    graph_kwargs = {'search_radius': 1,
                    'initialization_style': 'random'}

    # setup 2.3) signalling field (exosomes + cell-cell signalling via W matrix)
    # Note: consider rescale gamma as gamma / num_cells * num_plaquette
    # global gamma acts as field_strength_signal, it tunes exosomes AND sent field
    # TODO implement exosomes for dynamics_blockparallel case
    exosome_string = "no_exo_field"  # on/off/all/no_exo_field; 'off' = send info only 'off' genes
    exosome_remove_ratio = 0.0       # amount of exo field idx to randomly prune from each cell

    # setup 2.4) applied/manual field (part 1)
    # size [N x steps] or size [NM x steps] or None
    # field_applied = construct_app_field_from_genes(
    #    IPSC_EXTENDED_GENES_EFFECTS, simsetup['GENE_ID'], num_steps=steps)
    field_applied = None

    # setup 2.5) applied/manual field (part 2) add housekeeping field with strength kappa
    flag_housekeeping = False
    field_housekeeping_strength = 0.0  # aka Kappa
    assert not flag_housekeeping
    if flag_housekeeping:
        assert field_housekeeping_strength > 0
        # housekeeping auto (via model extension)
        field_housekeeping = np.zeros(simsetup_main['N'])
        if simsetup_main['K'] > 0:
            field_housekeeping[-simsetup_main['K']:] = 1.0
            print(field_applied)
        else:
            print('Note gene 0 (on), 1 (on), 2 (on) are HK in A1 memories')
            print('Note gene 4 (off), 5 (on) are HK in C1 memories')
            field_housekeeping[4] = 1.0
            field_housekeeping[5] = 1.0
        if field_applied is not None:
            field_applied += field_housekeeping_strength * field_housekeeping
        else:
            field_applied = field_housekeeping_strength * field_housekeeping
    else:
        field_housekeeping = None

    # setup 2.6) optionally load an initial state for the lattice
    load_manual_init = False
    init_state_path = None
    if load_manual_init:
        init_state_path = INPUT_FOLDER + os.sep + 'manual_graphstate' + os.sep + 'X_8.txt'
        print('Note: in main, loading init graph state from file...')

    # 2) prep args for Multicell class instantiation
    multicell_kwargs_base = {
        'run_basedir': multirun_name,
        'beta': beta,
        'total_steps': total_steps,
        'num_cells': num_cells,
        'flag_blockparallel': flag_blockparallel,
        'graph_style': graph_style,
        'graph_kwargs': graph_kwargs,
        'autocrine': autocrine,
        'gamma': gamma,
        'exosome_string': exosome_string,
        'exosome_remove_ratio': exosome_remove_ratio,
        'kappa': kappa,
        'field_applied': field_applied,
        'flag_housekeeping': flag_housekeeping,
        'flag_state_int': flag_state_int,
        'plot_period': plot_period,
        'init_state_path': init_state_path,
    }

    num_runs = 1000
    ensemble = 1  # currently un-used
    run_dirs = [''] * num_runs

    # note we pickle the first runs instance for later loading
    for i in range(num_runs):
        print("On run %d (%d total)" % (i, num_runs))
        multicell_kwargs_local = multicell_kwargs_base.copy()

        # 1) modify multicell kwargs for each run
        # (A) local seed
        seed = i
        multicell_kwargs_local['seed'] = seed
        # (B) local run_label
        multicell_kwargs_local['run_subdir'] = 's%d' % seed

        # 2) instantiate
        multicell = Multicell(simsetup_main, verbose=True, **multicell_kwargs_local)
        run_dirs[i] = multicell.io_dict['basedir']

        # 2.1) save full state to file for the first run (place in parent dir)
        if i == 0:
            ppath = multirun_path + os.sep + 'multicell_template.pkl'
            with open(ppath, 'wb') as fp:
                pickle.dump(multicell, fp)

        # 3) run sim
        multicell.simulation_fast_to_fp(no_datatdict=True, no_visualize=True)

    # aggregate data from multiple runs
    if aggregate_data:
        print('Aggregating data in %s' % multirun_path)
        aggregate_manyruns(multirun_path)
