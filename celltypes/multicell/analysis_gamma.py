import numpy as np
import os

from singlecell.singlecell_simsetup import singlecell_simsetup
from utils.file_io import RUNS_FOLDER


def scan_gamma_dynamics(J, W, state, coordnum=8, verbose=False):
    critgamma = None
    for gamma in np.linspace(0.001, 0.1, 10000):
        Js_internal = np.dot(J, state)
        state_01 = (state + np.ones_like(state)) / 2.0
        h_field_nbr = gamma * coordnum * np.dot(W, state_01)
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

    if destabilize_celltypes_gamma:
        random_mem = False
        random_W = False
        simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W)
        coordnum = 8  # num neighbours which signals are received from
        W = simsetup['FIELD_SEND']
        J = simsetup['J']
        celltypes = (simsetup['XI'][:, a] for a in range(simsetup['P']))
        print('Scanning for monotype destabilizing gamma (for coordination number %d)' % coordnum)
        for idx, celltype in enumerate(celltypes):
            critgamma = scan_gamma_dynamics(J, W, celltype, coordnum=coordnum, verbose=False)
            print(idx, simsetup['CELLTYPE_LABELS'][idx], critgamma)
