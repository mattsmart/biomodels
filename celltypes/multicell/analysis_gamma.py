import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from multicell.multicell_class import Multicell
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


def fullscan_gamma_bifurcation_candidates(
        multicell_kwargs, simsetup_base, anchored=True, verbose=True, dg=1e-1, gmax=20.0):
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
        multicell_kwargs: kwargs to form Multicell which is recreated for each gamma during the scan
        simsetup_base: simsetup dict template storing J, W, singlecell parameters
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

    SPEEDUP CHANGES:
    - only multicell instantiation before the loop (not during)
    - REQUIRES no varying of simsetup (J, W, N) in the loop - will need to refactor if this changes
    """

    # build gamma_space
    gamma_space = np.arange(0.0, gmax, dg)
    num_gamma = len(gamma_space)
    bifurcation_candidate_sequence = []

    # construct multicell_base from kwargs
    multicell_base = Multicell(simsetup_base, verbose=False, **multicell_kwargs)
    #init_cond = multicell_base.graph_state_arr[:, 0]

    def descend_to_fp(multicell):
        multicell.dynamics_full(
            flag_visualize=False, flag_datastore=False, flag_savestates=False,
            end_at_fp=True, verbose=False)
        current_step = multicell.current_step
        fp = multicell.graph_state_arr[:, current_step]
        return fp

    def check_still_fp(test_fp, J_multicell):
        A = test_fp
        B = np.sign(np.dot(J_multicell, test_fp))  # TODO if any sgn(0), then what?
        return np.array_equal(A, B)

    # prep: perform gradient descent on the init cond to get our (potentially anchored) fixed point
    init_fp = descend_to_fp(multicell_base)
    prev_fp = np.copy(init_fp)  # used for iterative comparisons

    # speedup:
    multicell_local = multicell_base  # change attributes on the fly

    for i, gamma in enumerate(gamma_space):
        if i % 200 == 0:
            print("Checking %d/%d (gamma=%.3f)" % (i, num_gamma, gamma))
        multicell_kwargs_local = multicell_kwargs.copy()
        multicell_kwargs_local['gamma'] = gamma

        # 1) Re-build Multicell for gamma
        # TODO needed to recreate Multicell each step? seems very slow, optimize
        J_multicell = multicell_base.build_J_multicell(gamma=gamma, plot=False)
        multicell_local.gamma = gamma
        multicell_local.matrix_J_multicell = J_multicell

        # 2) gradient descent to fixed point
        if anchored:
            multicell_local.simulation_reset(provided_init_state=init_fp)
            step_fp = descend_to_fp(multicell_local)
            fp_unchanged = np.array_equal(step_fp, prev_fp)
            prev_fp = step_fp
        else:
            fp_unchanged = check_still_fp(prev_fp, J_multicell)
            if not fp_unchanged:
                multicell_local.simulation_reset(provided_init_state=prev_fp)
                prev_fp = descend_to_fp(multicell_local)

        # 3) report a bifurcation whenever the fixed point moves
        if not fp_unchanged:
            if verbose:
                print('fixed point shift at gamma=%.3f' % gamma)
            bifurcation_candidate_sequence.append(gamma)

    return bifurcation_candidate_sequence, gamma_space, multicell_base


def plot_bifurcation_candidates(bifurcation_candidates, gamma_space, outdir, show=False):
    # plot type A
    x = np.arange(len(bifurcation_candidates))
    y = np.array(bifurcation_candidates)
    plt.scatter(x, y, marker='x')
    plt.xlabel(r'$n$')
    plt.ylabel(r'${\gamma}^*_n$')
    plt.savefig(outdir + os.sep + 'bifurc_A.jpg')
    if show:
        plt.show()

    # plot type B
    x = np.arange(len(bifurcation_candidates))
    y_construct = np.zeros(len(gamma_space))
    k = 0
    g0 = 0.0
    total_bifurcation_candidates = len(bifurcation_candidates)
    print(bifurcation_candidates)
    for i, gamma in enumerate(gamma_space):
        if bifurcation_candidates[k] > gamma:
            y_construct[i] = g0
        else:
            g0 = bifurcation_candidates[k]
            if k < total_bifurcation_candidates - 1:
                k += 1
            y_construct[i] = g0
        print(i, gamma, y_construct[i], k)
    plt.plot(gamma_space, y_construct, '--', c='k')
    plt.plot(gamma_space, gamma_space, '-.', c='k', alpha=0.5)
    plt.scatter(bifurcation_candidates, bifurcation_candidates, marker='o')
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'${\gamma}^*_n$ Transitions (n=%d)' % len(bifurcation_candidates))
    plt.savefig(outdir + os.sep + 'bifurc_B.jpg')
    if show:
        plt.show()


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
    W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_W15maze.txt'
    #W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_2018maze.txt'
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

        # 1) choose BASE simsetup (core singlecell params J, W)
        W_override_path = INPUT_FOLDER + os.sep + 'manual_WJ' + os.sep + 'simsetup_W_9_W15maze.txt'
        simsetup_base = singlecell_simsetup(
                unfolding=True, random_mem=False, random_W=False, curated=curated, housekeeping=0)
        if W_override_path is not None:
            print('Note: in main, overriding W from file...')
            explicit_W = np.loadtxt(W_override_path, delimiter=',')
            simsetup_base['FIELD_SEND'] = explicit_W

        # 2) choose BASE Multicell class parameters
        bifurcation_path = RUNS_FOLDER + os.sep + 'explore' + os.sep + 'bifurcation'
        num_cells = 10 ** 2
        autocrine = False
        graph_style = 'lattice_square'
        graph_kwargs = {'search_radius': 1,
                        'periodic': True,
                        'initialization_style': 'dual'}
        load_manual_init = False
        init_state_path = None
        if load_manual_init:
            print('Note: in main, loading init graph state from file...')
            init_state_path = INPUT_FOLDER + os.sep + 'manual_graphstate' + os.sep + 'X_8.txt'

        multicell_kwargs_base = {
            'seed': 0,
            'run_basedir': bifurcation_path,
            'beta': np.Inf,
            'total_steps': 500,
            'num_cells': num_cells,
            'flag_blockparallel': False,
            'graph_style': graph_style,
            'graph_kwargs': graph_kwargs,
            'autocrine': autocrine,
            'gamma': 0.0,
            'exosome_string': 'no_exo_field',
            'exosome_remove_ratio': 0.0,
            'kappa': 0.0,
            'field_applied': None,
            'flag_housekeeping': False,
            'flag_state_int': True,
            'plot_period': 1,
            'init_state_path': init_state_path,
        }

        dg = 1#0.01
        gmax = 1000.0
        anchored = True
        bifurcation_candidates, gamma_space, multicell = fullscan_gamma_bifurcation_candidates(
            multicell_kwargs_base, simsetup_base, anchored=anchored, verbose=True,
            dg=dg, gmax=gmax)
        outdir = multicell.io_dict['datadir']

        plot_bifurcation_candidates(bifurcation_candidates, gamma_space, outdir, show=True)

        fpath_x = outdir + os.sep + 'bifurcation_candidates.txt'
        fpath_gamma = outdir + os.sep + 'gamma_space.txt'
        np.savetxt(fpath_x, bifurcation_candidates, '%.4f')
        np.savetxt(fpath_gamma, gamma_space, '%.4f')
