"""
Conventions
- params is 7-vector of the form: params[0] -> alpha_plus
                                  params[1] -> alpha_minus
                                  params[2] -> mu
                                  params[3] -> a           (usually normalized to 1)
                                  params[4] -> b           (b = 1 - delta)
                                  params[5] -> c           (c = 1 + s)
                                  params[6] -> N           (float not int)
                                  params[7] -> v_x
                                  params[8] -> v_y         (typically 0)
                                  params[9] -> v_z         (typically 0)
                                  params[10] -> mu_base    (typically 0)
- if an element of params is specified as None then a bifurcation range will be be found and used
"""

import numpy as np
from random import random
from scipy.integrate import ode, odeint
from scipy.optimize import approx_fprime, fsolve
from sympy import Symbol, solve, re

from constants import PARAMS_ID, CSV_DATA_TYPES, SIM_METHODS_VALID, INIT_COND, TIME_START, TIME_END, NUM_STEPS
from params import Params


def ode_system_vector(init_cond, times, params):
    dxvec_dt = params.ode_system_vector(init_cond, times)
    return dxvec_dt


def system_vector_obj_ode(t_scalar, r_idx, params):
    return ode_system_vector(r_idx, t_scalar, params)


def ode_euler(init_cond, times, params):
    dt = times[1] - times[0]
    r = np.zeros((len(times), params.numstates))
    r[0] = np.array(init_cond)
    for idx, t in enumerate(times[:-1]):
        v = ode_system_vector(r[idx], None, params)
        r[idx+1] = r[idx] + np.array(v)*dt
    return r, times


def ode_rk4(init_cond, times, params):
    dt = times[1] - times[0]
    r = np.zeros((len(times), params.numstates))
    r[0] = np.array(init_cond)
    obj_ode = ode(system_vector_obj_ode, jac=None)
    obj_ode.set_initial_value(init_cond, times[0])
    obj_ode.set_f_params(params)
    obj_ode.set_integrator('dopri5')
    idx = 1
    while obj_ode.successful() and obj_ode.t < times[-1]:
        obj_ode.integrate(obj_ode.t + dt)
        r[idx] = np.array(obj_ode.y)
        idx += 1
    return r, times


def ode_libcall(init_cond, times, params):
    fn = ode_system_vector
    r = odeint(fn, init_cond, times, args=(params,))
    return r, times


def reaction_propensities(r, step, params, fpt_flag=False):
    rxn_prop = params.rxn_prop(r[step])
    if fpt_flag:
        laststate_idx = params.numstates - 1
        laststate = r[step][laststate_idx]
        rxn_prop.append(params.mu*laststate)  # special transition events for z1->z2 (extra mutation)
    return rxn_prop


def bisecting_rxn_search_iter(propensities, L, R, T, m=0):
    while L<=R:
        m = int(np.floor((L + R) / 2))
        if propensities[m] <= T:
            L=m+1
        else:
            R=m-1
    return m


def bisecting_rxn_search_recurse(propensities, L, R, T, m=0):
    if L > R:
        return m
    m = int(np.floor((L + R) / 2))
    if propensities[m] <= T:
        return bisecting_rxn_search_recurse(propensities, m+1, R, T, m=m)
    else:
        return bisecting_rxn_search_recurse(propensities, L, m-1, T, m=m)


def stoch_gillespie(init_cond, num_steps, params, fpt_flag=False):
    # There are 12 transitions to consider:
    # - 6 birth/death of the form x_n -> x_n+1, (x birth, x death, ...), label these 0 to 5
    # - 3 transitions of the form x_n -> x_n-1, (x->y, y->x, y->z), label these 6 to 8
    # - 3 transitions associated with immigration (vx, vy, vz), label these 9 to 11
    # - 1 transitions for x->z (rare), label this 12
    # Gillespie algorithm has indefinite timestep size so consider total step count as input (times input not used)
    # Notes on fpt_flag:
    # - if fpt_flag (first passage time) adds extra rxn propensity for transition from z1->z2
    # - return r[until fpt], times[until fpt]

    time = 0.0
    r = np.zeros((num_steps, params.numstates))
    times_stoch = np.zeros(num_steps)
    r[0, :] = np.array(init_cond, dtype=int)  # note stochastic sim operates on integer population counts
    update_dict = params.update_dict
    fpt_rxn_idx = len(update_dict.keys()) - 1  # always use last element as special FPT event
    fpt_event = False
    for step in xrange(num_steps-1):
        print r[step]
        r1 = random()  # used to determine time of next reaction
        r2 = random()  # used to partition the probabilities of each reaction
        # compute propensity functions (alpha) and the partitions for all 12 transitions
        alpha = reaction_propensities(r, step, params, fpt_flag=fpt_flag)
        alpha_partitions = np.zeros(len(alpha)+1)
        alpha_sum = 0.0
        for i in xrange(len(alpha)):
            alpha_sum += alpha[i]
            alpha_partitions[i + 1] = alpha_sum
        #alpha_partitions = alpha_partitions / alpha_sum  #rescale r2 instead to save cycles

        # find time to first reaction
        tau = np.log(1 / r1) / alpha_sum

        # BISECTING SEARCH METHOD (slower for small number of reactions)
        #r2_scaled = alpha_sum * r2
        #rxn_idx = bisecting_rxn_search(alpha_partitions, 0, len(alpha_partitions), r2_scaled)
        #pop_updates = update_dict[rxn_idx]
        #r[step + 1] = r[step] + pop_updates

        #DIRECT SEARCH METHOD (faster for 14 or fewer rxns so far)
        r2_scaled = alpha_sum*r2
        for rxn_idx in xrange(len(alpha)):
            if alpha_partitions[rxn_idx] <= r2_scaled < alpha_partitions[rxn_idx + 1]:  # i.e. rxn_idx has occurred
                pop_updates = update_dict[rxn_idx]
                r[step+1] = r[step] + pop_updates
                break

        time += tau
        times_stoch[step + 1] = time
        if rxn_idx == fpt_rxn_idx:
            assert fpt_flag                          # just in case, not much cost
            return r[:step+2, :], times_stoch[:step+2]  # end sim because fpt achieved
    if fpt_flag:  # if code gets here should recursively continue the simulation
        init_cond = r[-1]
        r_redo, times_stoch_redo = stoch_gillespie(init_cond, num_steps, params, fpt_flag=fpt_flag)
        times_stoch_redo_shifted = times_stoch_redo + times_stoch[-1]  # shift start time of new sim by last time
        return np.concatenate((r, r_redo)), np.concatenate((times_stoch, times_stoch_redo_shifted))
    return r, times_stoch


def simulate_dynamics_general(init_cond, times, params, method="libcall"):
    if method == "libcall":
        return ode_libcall(init_cond, times, params)
    elif method == "rk4":
        return ode_rk4(init_cond, times, params)
    elif method == "euler":
        return ode_euler(init_cond, times, params)
    elif method == "gillespie":
        return stoch_gillespie(init_cond, len(times), params)
    else:
        raise ValueError("method arg invalid, must be one of %s" % SIM_METHODS_VALID)


def fp_from_timeseries(r, sim_method, tol=0.001):
    fp_test = r[-1,:]
    fp_check = r[-2,:]
    if np.linalg.norm(fp_test - fp_check) <= tol:
        return fp_test
    elif sim_method == "gillespie" and np.linalg.norm(fp_test - r[-5,:]) <= 5:  # TODO exit condition for gillespie SS
        return fp_test
    else:
        raise ValueError("timeseries endpoint not a fixed point using dist tol: %.2f" % tol)


def bifurc_value(params, bifurc_name):
    """
    Note: assumes params contains at most one None parameter
    # TODO: implement mu_base (analysis)
    """
    assert params.system == "default"
    p = params
    assert p.mu_base <= 10e-10
    if p.b is not None:
        delta = 1 - p.b
    if p.c is not None:
        s = p.c - 1
    # assumes threshold_2 is stronger constraint, atm hardcode rearrange expression for bifurc param
    if bifurc_name == "bifurc_b":
        delta_val = p.alpha_minus * p.alpha_plus / (s + p.alpha_plus) - (s + p.alpha_minus + p.mu)
        bifurc_val = 1 - delta_val
        return bifurc_val
    elif bifurc_name == "bifurc_c":
        """
        -bifurcation in s = c - 1 occurs at rightmost root of a1 quadratic criterion in general
        -note a1 always has 2 roots for physical parameters
        -note linear a0 criterion has positive slope in s and IS the derivative of a1 wrt s 
         and so its root will always at the midpoint of the two a1 roots
        -need a1 and a0 both positive, since a0 not positive left of its sol and a1 not positive 
         between its roots this implies a1's rightmost root gives the bifurcation point
        """
        poly = np.array([1, delta + p.alpha_plus + p.alpha_minus + p.mu, p.alpha_plus * (delta + p.mu)])
        roots = np.roots(poly)
        s_val = np.max(roots)
        bifurc_val = 1 + s_val
        return bifurc_val
    elif bifurc_name == "bifurc_mu":
        """
        -expect bifurcation in mu to behave similarly to bifurcation in delta (b)
        -this is due to similar functional location in a0, a1 expressions
        """
        mu_option0 = p.alpha_minus * p.alpha_plus / (s + p.alpha_plus) - (s + p.alpha_minus + delta)
        mu_option1 = -(2*s + p.alpha_minus + delta + p.alpha_plus)
        return np.max([mu_option0, mu_option1])
    else:
        raise ValueError(bifurc_name + ' not valid bifurc ID')


def threshold_1(params):
    p = params
    assert p.system == "default"
    assert p.mu_base <= 10e-10
    delta = 1 - p.b
    s = p.c - 1
    return 2 * s + delta + p.alpha_plus + p.alpha_minus + p.mu


def threshold_2(params):
    p = params
    assert p.system == "default"
    assert p.mu_base <= 10e-10
    delta = 1 - p.b
    s = p.c - 1
    return (s + p.alpha_plus) * (s + delta + p.alpha_minus + p.mu) - p.alpha_minus * p.alpha_plus


def q_get(params, sign):
    p = params
    assert p.system == "default"
    assert p.mu_base <= 10e-10
    assert sign in [-1, +1]
    delta = 1 - p.b
    s = p.c - 1
    bterm = p.alpha_plus - p.alpha_minus - p.mu - delta
    return 0.5 / p.alpha_minus * (bterm + sign * np.sqrt(bterm ** 2 + 4 * p.alpha_minus * p.alpha_plus))


def fp_location_noflow(params):
    p = params
    assert p.system == "default"
    q1 = q_get(params, +1)
    q2 = q_get(params, -1)
    assert p.mu_base <= 10e-10
    delta = 1 - p.b
    s = p.c - 1
    conjugate_fps = [[0,0,0], [0,0,0]]
    for idx, q in enumerate([q1,q2]):
        xi = p.N * (s + p.alpha_plus - p.alpha_minus * q) / (s + (delta + s) * q)
        yi = q * xi
        zi = p.N - xi - yi
        conjugate_fps[idx] = [xi, yi, zi]
    return [[0, 0, p.N], conjugate_fps[0], conjugate_fps[1]]


def fp_location_sympy_system(params):
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    sym_x = Symbol("x")
    sym_y = Symbol("y")
    state_vec = [sym_x, sym_y, params.N - sym_x - sym_y]

    # TODO check that new method matches old method
    p = params.mod_copy(params.system_variants(state_vec, None))
    """ OLD METHOD
    if params.system == "feedback_z":
        z = N - sym_x - sym_y
        alpha_plus = alpha_plus * (1 + z**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO*N)**HILL_EXP))
        alpha_minus = alpha_minus * (HILLORIG_Z0_RATIO*N)**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO*N)**HILL_EXP)
    elif params.system == "feedback_yz":
        yz = N - sym_x
        alpha_plus = alpha_plus * (1 + yz**HILL_EXP / (yz**HILL_EXP + (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP))
        alpha_minus = alpha_minus * (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP / (yz**HILL_EXP + (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP)
    elif params.system == "feedback_mu_XZ_model":
        z = N - sym_x - sym_y
        alpha_plus = 0.0
        alpha_minus = 0.0
        mu_base = mu_base * (1 + MUBASE_MULTIPLIER * z**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO * N)**HILL_EXP))
    """

    VV = (p.v_x + p.v_y + p.v_z) / p.N
    xdot = (p.c-p.a)/p.N*sym_x**2 + (p.c-p.b)/p.N*sym_x*sym_y + (p.a-p.c-p.alpha_plus-p.mu_base-VV)*sym_x + p.alpha_minus*sym_y + p.v_x
    ydot = (p.c-p.b)/p.N*sym_y**2 + (p.c-p.a)/p.N*sym_x*sym_y + (p.b-p.c-p.alpha_minus-p.mu-VV)*sym_y + p.alpha_plus*sym_x + p.v_y
    eqns = (xdot, ydot)
    solution = solve(eqns)
    solution_list = [[0,0,0], [0,0,0], [0,0,0]]
    for i in xrange(3):
        x_i = float(re(solution[i][sym_x]))
        y_i = float(re(solution[i][sym_y]))
        solution_list[i] = [x_i, y_i, p.N - x_i - y_i]
    return solution_list


def fp_location_sympy_quartic(params):
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    sym_x = Symbol("x")
    sym_y = Symbol("y")
    state_vec = [sym_x, sym_y, params.N - sym_x - sym_y]

    # TODO check that new method matches old method
    p = params.mod_copy(params.system_variants(state_vec, None))
    """ OLD METHOD
    if params.system == "feedback_z":
        z = N - sym_x - sym_y
        alpha_plus = alpha_plus * (1 + z**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO*N)**HILL_EXP))
        alpha_minus = alpha_minus * (HILLORIG_Z0_RATIO*N)**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO*N)**HILL_EXP)
    elif params.system == "feedback_yz":
        yz = N - sym_x
        alpha_plus = alpha_plus * (1 + yz**HILL_EXP / (yz**HILL_EXP + (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP))
        alpha_minus = alpha_minus * (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP / (yz**HILL_EXP + (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP)
    elif params.system == "feedback_mu_XZ_model":
        z = N - sym_x - sym_y
        alpha_plus = 0.0
        alpha_minus = 0.0
        mu_base = mu_base * (1 + MUBASE_MULTIPLIER * z**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO * N)**HILL_EXP))
    """
    VV = (p.v_x+p.v_y+p.v_z)/p.N
    a0 = (p.c-p.a)/p.N
    a1 = 0.0
    b0 = 0.0
    b1 = (p.c-p.b)/p.N
    c0 = (p.c-p.b)/p.N
    c1 = (p.c-p.a)/p.N
    d0 = (p.a-p.c-p.alpha_plus-p.mu_base-VV)
    d1 = p.alpha_plus
    e0 = p.alpha_minus
    e1 = (p.b-p.c-p.alpha_minus-p.mu-VV)
    f0 = p.v_x
    f1 = p.v_y
    eqn = b1*(a0*sym_x**2 + d0*sym_x + f0)**2 - (c0*sym_x + e0)*(a0*sym_x**2 + d0*sym_x + f0)*(c1*sym_x + e1) + d1*sym_x + f1*(c0*sym_x + e0)**2
    solution = solve(eqn)
    solution_list = [[0,0,0], [0,0,0], [0,0,0]]
    for i in xrange(3):
        x_i = float(re(solution[i]))
        y_i = -(a0*x_i**2 + d0*x_i + f0) / (c0*x_i + e0)  # WARNING TODO: ensure this denom is nonzero
        solution_list[i] = [x_i, y_i, p.N - x_i - y_i]
    return solution_list


def fsolve_func(xvec_guess, params):  # TODO: faster if split into 3 fns w.o if else for feedback cases
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    x0, y0 = xvec_guess
    state_guess = [x0, y0, params.N - x0 - y0]
    p = params.mod_copy(params.system_variants(state_guess, None))
    VV = (p.v_x + p.v_y + p.v_z) / p.N
    xdot = (p.c-p.a)/p.N*x0**2 + (p.c-p.b)/p.N*x0*y0 + (p.a-p.c-p.alpha_plus-p.mu_base-VV)*x0 + p.alpha_minus*y0 + p.v_x
    ydot = (p.c-p.b)/p.N*y0**2 + (p.c-p.a)/p.N*x0*y0 + (p.b-p.c-p.alpha_minus-p.mu-VV)*y0 + p.alpha_plus*x0 + p.v_y
    return [xdot, ydot]


def fp_location_fsolve(params, check_near_traj_endpt=True, gridsteps=15, tol=10e-1):
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    N = params.N
    unique_solutions = []
    # first check for roots near trajectory endpoints (possible stable roots)
    if check_near_traj_endpt:
        init_cond = INIT_COND
        times = np.linspace(TIME_START, TIME_END, NUM_STEPS + 1)
        traj, _ = simulate_dynamics_general(init_cond, times, params, method="libcall")
        fp_guess = traj[-1][0:2]
        solution, infodict, _, _ = fsolve(fsolve_func, fp_guess, (params), full_output=True)
        if np.linalg.norm(infodict["fvec"]) <= 10e-3:  # only append actual roots (i.e. f(x)=0)
            unique_solutions.append([solution[0], solution[1], N - solution[0] - solution[1]])
    # grid search of solution space (positive simplex):
    for i in xrange(gridsteps):
        x_guess = N*i/float(gridsteps)
        for j in xrange(gridsteps-i):
            y_guess = N * i / float(gridsteps)
            # TODO: this returns jacobian estimate.. use it
            solution, infodict, _, _ = fsolve(fsolve_func, [x_guess, y_guess], (params), full_output=True)
            append_flag = True
            for k, usol in enumerate(unique_solutions):
                if np.abs(solution[0] - usol[0]) <= tol:   # only store unique roots from list of all roots
                    append_flag = False
                    break
            if append_flag:
                if np.linalg.norm(infodict["fvec"]) <= 10e-3:    # only append actual roots (i.e. f(x)=0)
                    unique_solutions.append([solution[0], solution[1], N - solution[0] - solution[1]])
    return unique_solutions


def fp_location_general(params, solver_fsolve=True, solver_fast=False, solver_explicit=False):
    # TODO: sympy solver often fails when feedback added in
    # TODO: cleanup the flags here
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    if solver_fsolve:
        return fp_location_fsolve(params)
    elif solver_explicit:
        assert params.system == "default"
        return fp_location_noflow(params)
    else:
        if solver_fast:
            return fp_location_sympy_quartic(params)
        else:
            return fp_location_sympy_system(params)


def jacobian_3d(params, fp):
    p = params
    assert p.system == "default"
    assert p.mu_base <= 10e-10
    M = np.array([[p.a - p.alpha_plus - p.mu_base, p.alpha_minus, 0],
                  [p.alpha_plus, p.b - p.alpha_minus - p.mu, 0],
                  [p.mu_base, p.mu, p.c]])
    x, y, z = fp
    diag = p.a*x + p.b*y + p.c*z + p.v_x + p.v_y + p.v_z
    r1 = [diag + x*p.a, x*p.b, x*p.c]
    r2 = [y*p.a, diag + y*p.b, y*p.c]
    r3 = [z*p.a, z*p.b, diag + z*p.c]
    return M - 1/p.N*np.array([r1,r2,r3])


def jacobian_numerical_2d(params, fp):
    # TODO: can use numdifftools jacobian function instead
    # TODO: move scope of func xdot etc up and use them both in func fsolve
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    def func_xdot(fp):
        x, y = fp[0], fp[1]
        state_vec = [x, y, params.N - x - y]
        p = params.mod_copy(params.system_variants(state_vec, None))
        VV = (p.v_x + p.v_y + p.v_z) / p.N
        return (p.c - p.a) / p.N * x ** 2 + (p.c - p.b) / p.N * x * y + (p.a - p.c - p.alpha_plus - p.mu_base - VV) * x \
               + p.alpha_minus * y + p.v_x
    def func_ydot(fp):
        x, y = fp[0], fp[1]
        state_vec = [x, y, params.N - x - y]
        p = params.mod_copy(params.system_variants(state_vec, None))
        VV = (p.v_x + p.v_y + p.v_z) / p.N
        return (p.c-p.b)/p.N*y**2 + (p.c-p.a)/p.N*x*y + (p.b-p.c-p.alpha_minus-p.mu-VV)*y + p.alpha_plus*x + p.v_y
    epsilon = 10e-4
    row_x = approx_fprime(fp, func_xdot, epsilon)
    row_y = approx_fprime(fp, func_ydot, epsilon)
    return np.array([row_x, row_y])


def is_stable(params, fp, method="numeric_2d"):
    if method == "numeric_2d":
        assert len(fp) == 2
        J = jacobian_numerical_2d(params, fp)
        eigenvalues, V = np.linalg.eig(J)
    elif method == "algebraic_3d":
        J = jacobian_3d(params, fp)
        eigenvalues, V = np.linalg.eig(J)
    else:
        raise ValueError("method must be 'numeric_2d' or 'algebraic_3d'")
    return all(eig < 0 for eig in eigenvalues)


def get_stable_fp(params):
    fp_locs = fp_location_general(params, solver_fsolve=True)
    fp_locs_stable = []
    for fp in fp_locs:
        if is_stable(params, fp[0:2], method="numeric_2d"):
            fp_locs_stable.append(fp)
            # eigs,V = np.linalg.eig(jacobian_numerical_2d(params, fp[0:2], ode_system))
            # print fp, eigs
    return fp_locs_stable


def get_physical_and_stable_fp(params, verbose=False):
    fp_locs = fp_location_general(params, solver_fsolve=True)
    fp_locs_physical_and_stable = []
    for fp in fp_locs:
        if all([val > -0.1 for val in fp]):
            if is_stable(params, fp[0:2], method="numeric_2d"):
                fp_locs_physical_and_stable.append(fp)
                #eigs,V = np.linalg.eig(jacobian_numerical_2d(params, fp[0:2], ode_system))
                #print fp, eigs

    if verbose:
        print "\nFP NOTES for b,c", params.b, params.c
        print "ALL FP: (%d)" % len(fp_locs)
        for fp in fp_locs:
            print fp
        print "PHYS/STABLE FP: (%d)" % len(fp_locs_physical_and_stable)
        for fp in fp_locs_physical_and_stable:
            print fp
        if len(fp_locs_physical_and_stable) == 0:
            print "WARNING: 0 phys and stable FP"
            init_cond = INIT_COND
            times = np.linspace(TIME_START, TIME_END, NUM_STEPS + 1)
            traj, _ = simulate_dynamics_general(init_cond, times, params, method="libcall")
            fp_guess = traj[-1][:]
            print "FP from traj at all-x is:"
            print fp_guess
            print "ode_system_vector at possible FP is"
            print params.ode_system_vector(fp_guess, None)

    return fp_locs_physical_and_stable
