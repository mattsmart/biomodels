import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell.singlecell_simsetup import singlecell_simsetup
from singlecell.singlecell_linalg import sorted_eig
from utils.file_io import RUNS_FOLDER, INPUT_FOLDER


def scan_gamma_dynamics(J, W, state, coordnum=8, verbose=False, use_01=False):
    critgamma = None
    if use_01:
        state = (state + np.ones_like(state)) / 2.0
    for gamma in np.linspace(0.001, 0.8, 10000):
        Js_internal = np.dot(J, state)
        h_field_nbr = gamma * coordnum * np.dot(W, state)
        updated_state = np.sign(Js_internal + h_field_nbr)
        if np.array_equal(updated_state, state):
            if verbose:
                print(gamma, True)
        else:
            if critgamma is None:
                critgamma = gamma
            if verbose:
                print(gamma, False)
    return critgamma


if __name__ == '__main__':
    destabilize_celltypes_gamma = True
    flag_plot_multicell_evals = False


    main_seed = 0 #np.random.randint(1e6)
    curated = True
    random_mem = False        # TODO incorporate seed in random XI in simsetup/curated
    random_W = True          # TODO incorporate seed in random W in simsetup/curated

    #W_override_path = None
    W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_maze.txt'
    #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_mazeAlt5.txt'
    simsetup = singlecell_simsetup(
        unfolding=True, random_mem=random_mem, random_W=random_W, curated=curated, housekeeping=0)
    if W_override_path is not None:
        print('Note: in main, overriding W from file...')
        explicit_W = np.loadtxt(W_override_path, delimiter=',')
        simsetup['FIELD_SEND'] = explicit_W
    print("simsetup checks:")
    print("\tsimsetup['N'],", simsetup['N'])
    print("\tsimsetup['P'],", simsetup['P'])

    if destabilize_celltypes_gamma:
        coordnum = 8  # num neighbours which signals are received from
        W = simsetup['FIELD_SEND']
        J = simsetup['J']
        celltypes = (simsetup['XI'][:, a] for a in range(simsetup['P']))
        print('Scanning for monotype destabilizing gamma (for coordination number %d)' % coordnum)
        for idx, celltype in enumerate(celltypes):
            critgamma = scan_gamma_dynamics(J, W, celltype, coordnum=coordnum, verbose=False)
            print(idx, simsetup['CELLTYPE_LABELS'][idx], critgamma)

    if flag_plot_multicell_evals:
        # TODO implement or take from ipynb
        J_multicell = 1
        evals, evecs = sorted_eig(J_multicell)
