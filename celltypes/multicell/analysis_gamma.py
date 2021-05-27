import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from multicell.multicell import Multicell
from singlecell.singlecell_simsetup import singlecell_simsetup
from singlecell.singlecell_linalg import sorted_eig
from utils.file_io import RUNS_FOLDER, INPUT_FOLDER


def scan_plaquette_gamma_dynamics(J, W, state, coordnum=8, verbose=False, use_01=False):
    critgamma = None

    def get_state_send(state_send):
        if use_01:
            state_send = (state_send + np.ones_like(state_send)) / 2.0
        return state_send

    for gamma in np.linspace(0.001, 0.8, 10000):
        Js_internal = np.dot(J, state)

        # conditional 01 state send
        state_send = get_state_send(state)
        h_field_nbr = gamma * coordnum * np.dot(W, state_send)

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


def fullscan_gamma_bifurcation_candidates(multicell_template, init_cond, anchored=True, verbose=True):
    """
    For fixed initial condition and multicell parameters, slowly vary gamma.
    Find {gamma*}, the observed points where the fixed point has changed.
    The fixed point shifting is not a symptom of a bifurcation, for example consider pitchfork
     bifurcation, the two fixed points continue to shift after the (singular) bifurcation.
    TODO:
        Consider continuous dynamical system.
        When a static FP suddenly starts shifting in almost continuous fashion, that's a signature
         of a bifurcation (e.g. transcritical or pitchfork).
        What, if any, is the discrete (discrete time AND discrete state) analog of this?
    Args:
        multicell_template: Multicell which is recreated for each gamma during the scan
        init_cond: initial state of the multicell graph
        anchored: if True, use a fixed initial condition for each gradient descent;
                  else will use the previous fixed point as the initial condition
    Returns:
         list: the sequence of points {gamma*_n} where bifurcations have occurred
    Notes:
        - naively can do full gradient descent to reach each fixed point
        - once a fixed point is found, there is a faster way to check that it remains a fixed point:
          simply check that s* = sgn(J_multicell s*)   -- this is useful in "not anchored" case
        - if this vector condition holds then the FP is unchanged; when it breaks there is a
          bifurcation point (which is recorded) and the new FP should be found via descent
    """

    # build gamma_space
    dy = 1e-3
    gamma_max = 20.0
    gamma_space = np.arange(0.0, gamma_max, dy)
    bifurcation_candidate_sequence = []

    def descend_to_fp(init_state):
        # TODO
        return fp

    def check_still_fp(test_fp, J_multicell):
        A = test_fp
        B = np.sign(np.dot(J_multicell, test_fp))  # TODO if any sgn(0), then what?
        return A == B

    def J_multicell_given_gamma(gamma):
        return multicell_template.build_J_multicell(gamma=gamma, plot=False)

    # 0) loop preparations
    # first: add init cond to multicell template
    # TODO maybe
    # perform gradient descent on the init cond to get our (potentially anchored) fixed point
    init_fp = descend_to_fp(init_cond)
    prev_fp = np.copy(init_fp)  # used for iterative comparisons

    for gamma in gamma_space:

        # 1) Re-build Multicell for gamma (TODO optimize)
        J_multicell = J_multicell_given_gamma(gamma)
        multicell_local = ... foo(multicell_template)  # TODO needed?

        # 2) gradient descent to fixed point
        if anchored:
            step_fp = descend_to_fp(init_fp)
            fp_unchanged = (step_fp == prev_fp)
            prev_fp = step_fp
        else:
            fp_unchanged = check_still_fp(prev_fp, J_multicell)
            if not fp_unchanged:
                prev_fp = descend_to_fp(prev_fp)

        # 3) report a bifurcation whenever the fixed point moves
        if not fp_unchanged:
            if verbose:
                print('bifurcation at gamma=%.3f' % gamma)
            bifurcation_candidate_sequence.append(gamma)

    return bifurcation_candidate_sequence


# TODO check that not symmetrizing loaded W (in simsetup)
if __name__ == '__main__':
    force_symmetry_W = True
    destabilize_celltypes_gamma = False
    flag_plot_multicell_evals = False
    flag_bifurcation_sequence = True


    main_seed = 0 #np.random.randint(1e6)
    curated = True
    random_mem = False        # TODO incorporate seed in random XI in simsetup/curated
    random_W = False          # TODO incorporate seed in random W in simsetup/curated

    #W_override_path = None
    W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_maze.txt'
    #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_2018maze.txt'
    #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'matrix_W_9_W15maze.txt'
    #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_random1.txt'
    simsetup = singlecell_simsetup(
        unfolding=True, random_mem=random_mem, random_W=random_W, curated=curated, housekeeping=0)
    if W_override_path is not None:
        print('Note: in main, overriding W from file...')
        explicit_W = np.loadtxt(W_override_path, delimiter=',')
        simsetup['FIELD_SEND'] = explicit_W
    print("simsetup checks:")
    print("\tsimsetup['N'],", simsetup['N'])
    print("\tsimsetup['P'],", simsetup['P'])

    if force_symmetry_W:
        W = simsetup['FIELD_SEND']
        # V1: take simple sym
        #simsetup['FIELD_SEND'] = (W + W.T)/2
        # V2: take upper triangular part
        Wdiag = np.diag(np.diag(W))
        Wut = np.triu(W, 1)
        simsetup['FIELD_SEND'] = Wut + Wut.T + Wdiag
        # V3: take lower triangular part
        #Wdiag = np.diag(np.diag(W))1
        #Wut = np.tril(W, -1)
        #simsetup['FIELD_SEND'] = Wut + Wut.T + Wdiag
        # Save symmetrized W
        np.savetxt('Wsym.txt', simsetup['FIELD_SEND'], '%.4f', delimiter=',')
    print(simsetup['FIELD_SEND'])

    if destabilize_celltypes_gamma:
        coordnum = 8  # num neighbours which signals are received from
        W = simsetup['FIELD_SEND']
        J = simsetup['J']
        celltypes = (simsetup['XI'][:, a] for a in range(simsetup['P']))
        print('Scanning for monotype destabilizing gamma (for coordination number %d)' % coordnum)
        for idx, celltype in enumerate(celltypes):
            critgamma = scan_plaquette_gamma_dynamics(J, W, celltype, coordnum=coordnum, verbose=False)
            print(idx, simsetup['CELLTYPE_LABELS'][idx], critgamma)

    if flag_plot_multicell_evals:
        # TODO implement or take from ipynb
        J_multicell = 1
        evals, evecs = sorted_eig(J_multicell)

    if flag_bifurcation_sequence:

        # 1) access specific manyrun (gives access to multicell pkl file)
        manyruns_label = 'Wmaze15_R5_gamma20.00_10k_p3_M100'
        manyruns_dirpath = RUNS_FOLDER + os.sep + 'multicell_manyruns' + os.sep + manyruns_label
        manyruns_pkl_choice = manyruns_dirpath + os.sep + 'multicell_template.pkl'

        multicell_pkl_path = manyruns_pkl_choice
        with open(multicell_pkl_path, 'rb') as pickle_file:
            multicell_template = pickle.load(pickle_file)  # unpickling multicell object

        # 2) select intiial condition
        init_cond = ...  # TODO

        # 2) construct gamma range
        gamma_vals = np.arange(0.0, gamma_max + dy, dy)


        fullscan_gamma_bifurcations(multicell_template, init_cond, anchored=True, verbose=True)