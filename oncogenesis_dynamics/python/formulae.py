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

import csv
import numpy as np
from os import sep
from random import random
from scipy.integrate import ode, odeint
from scipy.optimize import approx_fprime, fsolve
from sympy import Symbol, solve, re

import trajectory
from constants import PARAMS_ID, CSV_DATA_TYPES, SIM_METHODS, PARAM_Z0_RATIO, PARAM_Y0_PLUS_Z0_RATIO, PARAM_HILL, \
                      ODE_SYSTEMS, PARAMS_ID_INV, PARAM_GAMMA
from params import Params


def system_variants(init_cond, times, params):
    x, y, z = init_cond
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params.params_list()

    if params.system == "feedback_z":
        alpha_plus = alpha_plus * (1 + z**PARAM_HILL / (z**PARAM_HILL + (PARAM_Z0_RATIO*N)**PARAM_HILL))
        alpha_minus = alpha_minus * (PARAM_Z0_RATIO*N)**PARAM_HILL / (z**PARAM_HILL + (PARAM_Z0_RATIO*N)**PARAM_HILL)
    elif params.system == "feedback_yz":
        yz = y + z
        alpha_plus = alpha_plus * (1 + yz**PARAM_HILL / (yz**PARAM_HILL + (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL))
        alpha_minus = alpha_minus * (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL / (yz**PARAM_HILL + (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL)
    elif params.system == "feedback_mu_XZ_model":
        alpha_plus = 0.0
        alpha_minus = 0.0
        mu_base = mu_base * (1 + PARAM_GAMMA * z**PARAM_HILL / (z**PARAM_HILL + (PARAM_Z0_RATIO * N)**PARAM_HILL))

    return alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base


def ode_system_vector(init_cond, times, params):
    x, y, z = init_cond
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = system_variants(init_cond, times, params)
    fbar = (a * x + b * y + c * z + v_x + v_y + v_z) / N
    dxdt = v_x - x * (alpha_plus + mu_base) + y * alpha_minus + (a - fbar) * x
    dydt = v_y + x * alpha_plus - y * (alpha_minus + mu) + (b - fbar) * y
    dzdt = v_z + y * mu + z*mu_base + (c - fbar) * z
    return [dxdt, dydt, dzdt]


def system_vector_obj_ode(t_scalar, r_idx, params):
    return ode_system_vector(r_idx, t_scalar, params)


def ode_euler(init_cond, times, params):
    dt = times[1] - times[0]
    r = np.zeros((len(times), 3))
    r[0] = np.array(init_cond)
    for idx, t in enumerate(times[:-1]):
        v = ode_system_vector(r[idx], None, params)
        r[idx+1] = r[idx] + np.array(v)*dt
    return r, times


def ode_rk4(init_cond, times, params):
    dt = times[1] - times[0]
    r = np.zeros((len(times), 3))
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
    r = odeint(fn, init_cond, times, args=tuple(params))
    return r, times


def reaction_propensities(r, step, params, fpt_flag=False):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = system_variants(r[step], None, params)
    x_n, y_n, z_n = r[step]
    fbar = (a*x_n + b*y_n + c*z_n + v_x + v_y + v_z) / N    # TODO flag to switch N to x + y + z
    rxn_prop = [a*x_n, fbar*(x_n),                      # birth/death events for x  TODO: is it fbar*(x_n - 1)
                b*y_n, fbar*(y_n),                      # birth/death events for y  TODO: is it fbar*(y_n - 1)
                c*z_n, fbar*(z_n),                      # birth/death events for z  TODO: is it fbar*(z_n - 1)
                alpha_plus*x_n, alpha_minus*y_n, mu*y_n,    # transition events
                v_x, v_y, v_z,                              # immigration events  #TODO maybe wrong
                mu_base*x_n]                                # special transition events (x->z)
    if fpt_flag:
        rxn_prop.append(mu*z_n)                             # special transition events for z1->z2 (extra mutation)
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
    r = np.zeros((num_steps, 3))
    times_stoch = np.zeros(num_steps)
    r[0, :] = np.array(init_cond, dtype=int)  # note stochastic sim operates on integer population counts
    update_dict = {0: [1, 0, 0], 1: [-1, 0, 0],                  # birth/death events for x
                   2: [0, 1, 0], 3: [0, -1, 0],                  # birth/death events for y
                   4: [0, 0, 1], 5: [0, 0, -1],                  # birth/death events for z
                   6: [-1, 1, 0], 7: [1, -1, 0], 8: [0, -1, 1],  # transition events
                   9: [1, 0, 0], 10: [0, 1, 0], 11: [0, 0, 1],   # immigration events
                   12: [-1, 0, 1], 13: [0, 0, -1]}               # special x->z, fpt z1->z2 (z2 untracked) transitions
    fpt_rxn_idx = 13
    fpt_event = False
    for step in xrange(num_steps-1):
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
        raise ValueError("method arg invalid, must be one of %s" % SIM_METHODS)


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
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params.params_list()
    assert mu_base <= 10e-10
    if b is not None:
        delta = 1 - b
    if c is not None:
        s = c - 1
    # assumes threshold_2 is stronger constraint, atm hardcode rearrange expression for bifurc param
    if bifurc_name == "bifurc_b":
        delta_val = alpha_minus * alpha_plus / (s + alpha_plus) - (s + alpha_minus + mu)
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
        p = np.array([1, delta + alpha_plus + alpha_minus + mu, alpha_plus * (delta + mu)])
        roots = np.roots(p)
        s_val = np.max(roots)
        bifurc_val = 1 + s_val
        return bifurc_val
    elif bifurc_name == "bifurc_mu":
        """
        -expect bifurcation in mu to behave similarly to bifurcation in delta (b)
        -this is due to similar functional location in a0, a1 expressions
        """
        mu_option0 = alpha_minus * alpha_plus / (s + alpha_plus) - (s + alpha_minus + delta)
        mu_option1 = -(2*s + alpha_minus + delta + alpha_plus)
        return np.max([mu_option0, mu_option1])
    else:
        raise ValueError(bifurc_name + ' not valid bifurc ID')


def threshold_1(params):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params.param_list()
    assert mu_base <= 10e-10
    delta = 1 - b
    s = c - 1
    return 2 * s + delta + alpha_plus + alpha_minus + mu


def threshold_2(params):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params.param_list()
    assert mu_base <= 10e-10
    delta = 1 - b
    s = c - 1
    return (s + alpha_plus) * (s + delta + alpha_minus + mu) - alpha_minus * alpha_plus


def q_get(params, sign):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params.param_list()
    assert mu_base <= 10e-10
    assert sign in [-1, +1]
    delta = 1 - b
    s = c - 1
    bterm = alpha_plus - alpha_minus - mu - delta
    return 0.5 / alpha_minus * (bterm + sign * np.sqrt(bterm ** 2 + 4 * alpha_minus * alpha_plus))


def fp_location_noflow(params):
    q1 = q_get(params, +1)
    q2 = q_get(params, -1)
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params.param_list()
    assert mu_base <= 10e-10
    delta = 1 - b
    s = c - 1
    conjugate_fps = [[0,0,0], [0,0,0]]
    for idx, q in enumerate([q1,q2]):
        xi = N * (s + alpha_plus - alpha_minus * q) / (s + (delta + s) * q)
        yi = q * xi
        zi = N - xi - yi
        conjugate_fps[idx] = [xi, yi, zi]
    return [[0, 0, N], conjugate_fps[0], conjugate_fps[1]]


def fp_location_sympy_system(params):
    sym_x = Symbol("x")
    sym_y = Symbol("y")
    state_vec = [sym_x, sym_y, params.N - sym_x - sym_y]

    # TODO check that new method matches old method
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = system_variants(state_vec, None, params)
    """ OLD METHOD
    if params.system == "feedback_z":
        z = N - sym_x - sym_y
        alpha_plus = alpha_plus * (1 + z**PARAM_HILL / (z**PARAM_HILL + (PARAM_Z0_RATIO*N)**PARAM_HILL))
        alpha_minus = alpha_minus * (PARAM_Z0_RATIO*N)**PARAM_HILL / (z**PARAM_HILL + (PARAM_Z0_RATIO*N)**PARAM_HILL)
    elif params.system == "feedback_yz":
        yz = N - sym_x
        alpha_plus = alpha_plus * (1 + yz**PARAM_HILL / (yz**PARAM_HILL + (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL))
        alpha_minus = alpha_minus * (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL / (yz**PARAM_HILL + (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL)
    elif params.system == "feedback_mu_XZ_model":
        z = N - sym_x - sym_y
        alpha_plus = 0.0
        alpha_minus = 0.0
        mu_base = mu_base * (1 + PARAM_GAMMA * z**PARAM_HILL / (z**PARAM_HILL + (PARAM_Z0_RATIO * N)**PARAM_HILL))
    """

    VV = (v_x + v_y + v_z) / N
    xdot = (c-a)/N*sym_x**2 + (c-b)/N*sym_x*sym_y + (a-c-alpha_plus-mu_base-VV)*sym_x + alpha_minus*sym_y + v_x
    ydot = (c-b)/N*sym_y**2 + (c-a)/N*sym_x*sym_y + (b-c-alpha_minus-mu-VV)*sym_y + alpha_plus*sym_x + v_y
    eqns = (xdot, ydot)
    solution = solve(eqns)
    solution_list = [[0,0,0], [0,0,0], [0,0,0]]
    for i in xrange(3):
        x_i = float(re(solution[i][sym_x]))
        y_i = float(re(solution[i][sym_y]))
        solution_list[i] = [x_i, y_i, N - x_i - y_i]
    return solution_list


def fp_location_sympy_quartic(params):
    sym_x = Symbol("x")
    sym_y = Symbol("y")
    state_vec = [sym_x, sym_y, params.N - sym_x - sym_y]

    # TODO check that new method matches old method
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = system_variants(state_vec, None, params)
    """ OLD METHOD
    if params.system == "feedback_z":
        z = N - sym_x - sym_y
        alpha_plus = alpha_plus * (1 + z**PARAM_HILL / (z**PARAM_HILL + (PARAM_Z0_RATIO*N)**PARAM_HILL))
        alpha_minus = alpha_minus * (PARAM_Z0_RATIO*N)**PARAM_HILL / (z**PARAM_HILL + (PARAM_Z0_RATIO*N)**PARAM_HILL)
    elif params.system == "feedback_yz":
        yz = N - sym_x
        alpha_plus = alpha_plus * (1 + yz**PARAM_HILL / (yz**PARAM_HILL + (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL))
        alpha_minus = alpha_minus * (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL / (yz**PARAM_HILL + (PARAM_Y0_PLUS_Z0_RATIO*N)**PARAM_HILL)
    elif params.system == "feedback_mu_XZ_model":
        z = N - sym_x - sym_y
        alpha_plus = 0.0
        alpha_minus = 0.0
        mu_base = mu_base * (1 + PARAM_GAMMA * z**PARAM_HILL / (z**PARAM_HILL + (PARAM_Z0_RATIO * N)**PARAM_HILL))
    """
    VV = (v_x+v_y+v_z)/N
    a0 = (c-a)/N
    a1 = 0.0
    b0 = 0.0
    b1 = (c-b)/N
    c0 = (c-b)/N
    c1 = (c-a)/N
    d0 = (a-c-alpha_plus-mu_base-VV)
    d1 = alpha_plus
    e0 = alpha_minus
    e1 = (b-c-alpha_minus-mu-VV)
    f0 = v_x
    f1 = v_y
    eqn = b1*(a0*sym_x**2 + d0*sym_x + f0)**2 - (c0*sym_x + e0)*(a0*sym_x**2 + d0*sym_x + f0)*(c1*sym_x + e1) + d1*sym_x + f1*(c0*sym_x + e0)**2
    solution = solve(eqn)
    solution_list = [[0,0,0], [0,0,0], [0,0,0]]
    for i in xrange(3):
        x_i = float(re(solution[i]))
        y_i = -(a0*x_i**2 + d0*x_i + f0) / (c0*x_i + e0)  # WARNING TODO: ensure this denom is nonzero
        solution_list[i] = [x_i, y_i, N - x_i - y_i]
    return solution_list


def fsolve_func(xvec_guess, params):  # TODO: faster if split into 3 fns w.o if else for feedback cases
    x0, y0 = xvec_guess
    state_guess = [x0, y0, params.N - x0 - y0]
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = system_variants(state_guess, None, params)
    VV = (v_x + v_y + v_z) / N
    xdot = (c-a)/N*x0**2 + (c-b)/N*x0*y0 + (a-c-alpha_plus-mu_base-VV)*x0 + alpha_minus*y0 + v_x
    ydot = (c-b)/N*y0**2 + (c-a)/N*x0*y0 + (b-c-alpha_minus-mu-VV)*y0 + alpha_plus*x0 + v_y
    return [xdot, ydot]


def fp_location_fsolve(params, check_near_traj_endpt=True, gridsteps=15, tol=10e-1):
    N = params.N
    unique_solutions = []
    # first check for roots near trajectory endpoints (possible stable roots)
    if check_near_traj_endpt:
        traj, _, _, _ = trajectory.trajectory_simulate(params, flag_showplt=False, flag_saveplt=False)
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
    assert params.system == "default"
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params.params_list()
    assert mu_base <= 10e-10
    M = np.array([[a - alpha_plus - mu_base, alpha_minus, 0],
                  [alpha_plus, b - alpha_minus - mu, 0],
                  [mu_base, mu, c]])
    x, y, z = fp
    diag = a*x + b*y + c*z + v_x + v_y + v_z
    r1 = [diag + x*a, x*b, x*c]
    r2 = [y*a, diag + y*b, y*c]
    r3 = [z*a, z*b, diag + z*c]
    return M - 1/N*np.array([r1,r2,r3])


def jacobian_numerical_2d(params, fp):
    # TODO: can use numdifftools jacobian function instead
    # TODO: move scope of func xdot etc up and use them both in func fsolve
    def func_xdot(fp):
        x, y = fp[0], fp[1]
        state_vec = [x, y, params.N - x - y]
        alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = system_variants(state_vec, None, params)
        VV = (v_x + v_y + v_z) / N
        return (c - a) / N * x ** 2 + (c - b) / N * x * y + (a - c - alpha_plus - mu_base - VV) * x + alpha_minus * y + v_x
    def func_ydot(fp):
        x, y = fp[0], fp[1]
        state_vec = [x, y, params.N - x - y]
        alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = system_variants(state_vec, None, params)
        VV = (v_x + v_y + v_z) / N
        return (c-b)/N*y**2 + (c-a)/N*x*y + (b-c-alpha_minus-mu-VV)*y + alpha_plus*x + v_y
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


def get_physical_and_stable_fp(params):
    fp_locs = fp_location_general(params, solver_fsolve=True)
    fp_locs_physical_and_stable = []
    for fp in fp_locs:
        if all([val > -0.1 for val in fp]):
            if is_stable(params, fp[0:2], method="numeric_2d"):
                fp_locs_physical_and_stable.append(fp)
                #eigs,V = np.linalg.eig(jacobian_numerical_2d(params, fp[0:2], ode_system))
                #print fp, eigs
    return fp_locs_physical_and_stable
