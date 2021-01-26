import numpy as np
import os

from multicell.multicell_simulate import Multicell
from singlecell.singlecell_simsetup import singlecell_simsetup
from utils.file_io import INPUT_FOLDER


if __name__ == '__main__':

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
    total_steps = 40           # global NUM_LATTICE_STEPS
    plot_period = 1
    flag_state_int = True
    flag_blockparallel = False
    beta = 2000.0
    gamma = 20.0               # i.e. field_signal_strength
    kappa = 0.0                # i.e. field_applied_strength

    # setup 2.2) graph options
    autocrine = False
    graph_style = 'lattice_square'
    graph_kwargs = {'search_radius': 1,
                    'initialization_style': 'dual'}

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

    # 2) prep args for Multicell class instantiation
    multicell_kwargs_base = {
        'run_basedir': 'multicell_manyruns',
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
    }

    num_runs = 10
    ensemble = 1  # currently un-used
    for i in range(num_runs):
        print("On run %d (%d total)" % (i, num_runs))
        multicell_kwargs_local = multicell_kwargs_base.copy()

        # 1) modify multicell kwargs for each run
        # (A) local seed
        seed = i
        multicell_kwargs_local['seed'] = i
        # (B) local run_label
        multicell_kwargs_local['run_subdir'] = 's%d' % seed

        # 2) instantiate
        multicell = Multicell(simsetup_main, verbose=True, **multicell_kwargs_local)

        # 3) run sim
        multicell.simulation_fast_to_fp()
