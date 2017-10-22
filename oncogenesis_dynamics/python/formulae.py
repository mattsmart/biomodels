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
                      ODE_SYSTEMS, PARAMS_ID_INV


def system_vector(init_cond, times, system, alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z):
    x, y, z = init_cond
    fbar = (a * x + b * y + c * z + v_x + v_y + v_z) / N
    if system == "feedback_z":
        alpha_plus = alpha_plus * (1 + z / (z + PARAM_Z0_RATIO*N))
        alpha_minus = alpha_minus * PARAM_Z0_RATIO*N / (z + PARAM_Z0_RATIO*N)
    elif system == "feedback_yz":
        yz = y + z
        alpha_plus = alpha_plus * (1 + yz / (yz + PARAM_Y0_PLUS_Z0_RATIO * N))
        alpha_minus = alpha_minus * PARAM_Y0_PLUS_Z0_RATIO * N / (yz + PARAM_Y0_PLUS_Z0_RATIO * N)
    dxdt = v_x - x * alpha_plus + y * alpha_minus + (a - fbar) * x
    dydt = v_y + x * alpha_plus - y * (alpha_minus + mu) + (b - fbar) * y
    dzdt = v_z + y * mu + (c - fbar) * z
    return [dxdt, dydt, dzdt]


def system_vector_obj_ode(t_scalar, r_idx, system, alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z):
    return system_vector(r_idx, t_scalar, system, alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z)


def ode_euler(init_cond, times, system, params):
    fn = system_vector
    odeparams = [system] + params
    dt = times[1] - times[0]
    r = np.zeros((len(times), 3))
    r[0] = np.array(init_cond)
    for idx, t in enumerate(times[:-1]):
        v = fn(r[idx], None, *odeparams)
        r[idx+1] = r[idx] + np.array(v)*dt
    return r, times


def ode_rk4(init_cond, times, system, params):
    dt = times[1] - times[0]
    r = np.zeros((len(times), 3))
    r[0] = np.array(init_cond)
    odeparams = [system] + params
    fn = system_vector_obj_ode
    obj_ode = ode(fn, jac=None)
    obj_ode.set_initial_value(init_cond, times[0])
    obj_ode.set_f_params(*odeparams)
    obj_ode.set_integrator('dopri5')
    idx = 1
    while obj_ode.successful() and obj_ode.t < times[-1]:
        obj_ode.integrate(obj_ode.t + dt)
        r[idx] = np.array(obj_ode.y)
        idx += 1
    return r, times


def ode_libcall(init_cond, times, system, params):
    fn = system_vector
    odeparams = [system] + params
    r = odeint(fn, init_cond, times, args=tuple(odeparams))
    return r, times


def reaction_propensities(r, step, system, params):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    x_n, y_n, z_n = r[step]
    if system == "feedback_z":
        alpha_plus = alpha_plus * (1 + z_n / (z_n + PARAM_Z0_RATIO * N))
        alpha_minus = alpha_minus * PARAM_Z0_RATIO * N / (z_n + PARAM_Z0_RATIO * N)
    elif system == "feedback_yz":
        yz = y_n + z_n
        alpha_plus = alpha_plus * (1 + yz / (yz + PARAM_Y0_PLUS_Z0_RATIO * N))
        alpha_minus = alpha_minus * PARAM_Y0_PLUS_Z0_RATIO * N / (yz + PARAM_Y0_PLUS_Z0_RATIO * N)
    fbar = (a*x_n + b*y_n + c*z_n + v_x + v_y + v_z) / N  # TODO flag to switch N to x + y + z
    return [a*x_n, fbar*(x_n - 1),                      # birth/death events for x
            b*y_n, fbar*(y_n - 1),                      # birth/death events for y
            c*z_n, fbar*(z_n - 1),                      # birth/death events for z
            alpha_plus*x_n, alpha_minus*y_n, mu*y_n,    # transition events
            v_x, v_y, v_z]                              # immigration events  #TODO maybe wrong


def stoch_gillespie(init_cond, times, system, params):
    # There are 12 transitions to consider:
    # - 6 birth/death of the form x_n -> x_n+1, (x birth, x death, ...), label these 0 to 5
    # - 3 transitions of the form x_n -> x_n+1, (x->y, y->x, y->z), label these 6 to 8
    # - 3 transitions associated with immigration (vx, vy, vz), label these 9 to 11
    # Gillespie algorithm has indefinite timestep size so consider total step count as input (times input not used)

    total_steps = len(times)
    #dt = times[1] - times[0]
    r = np.zeros((total_steps, 3))
    time = times[0]
    times_stoch = np.zeros(total_steps)
    r[0] = np.array(init_cond, dtype=int)  # note stochastic sim operates on integer population counts
    update_dict = {0: [1, 0, 0], 1: [-1, 0, 0],                  # birth/death events for x
                   2: [0, 1, 0], 3: [0, -1, 0],                  # birth/death events for y
                   4: [0, 0, 1], 5: [0, 0, -1],                  # birth/death events for z
                   6: [-1, 1, 0], 7: [1, -1, 0], 8: [0, -1, 1],  # transition events
                   9: [1, 0, 0], 10: [0, 1, 0], 11: [0, 0, 1]}   # immigration events
    for step in xrange(total_steps-1):
        r1 = random()  # used to determine time of next reaction
        r2 = random()  # used to partition the probabilities of each reaction
        # compute propensity functions (alpha) and the partitions for all 12 transitions
        alpha = reaction_propensities(r, step, system, params)
        alpha_partitions = np.zeros(len(alpha)+1)
        alpha_sum = 0.0
        for i in xrange(len(alpha)):
            alpha_sum += alpha[i]
            alpha_partitions[i + 1] = alpha_sum
        alpha_partitions = alpha_partitions / alpha_sum
        # find time to first reaction
        tau = np.log(1 / r1) / alpha_sum
        # compute number of molecules at time t + tau
        for species in xrange(3):
            for rxn_idx in xrange(len(alpha)):
                if alpha_partitions[rxn_idx] <= r2 < alpha_partitions[rxn_idx + 1]:  # i.e. rxn idx has occurred
                    pop_updates = update_dict[rxn_idx]
                    r[step+1] = r[step] + pop_updates
        time += tau
        times_stoch[step + 1] = time
        #if i % 10 == 0:
        #    print step
    return r, times_stoch


def simulate_dynamics_general(init_cond, times, params, method="libcall", system="default"):
    assert system in ODE_SYSTEMS
    if method == "libcall":
        return ode_libcall(init_cond, times, system, params)
    elif method == "rk4":
        return ode_rk4(init_cond, times, system, params)
    elif method == "euler":
        return ode_euler(init_cond, times, system, params)
    elif method == "gillespie":
        return stoch_gillespie(init_cond, times, system, params)
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
    """
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
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
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    delta = 1 - b
    s = c - 1
    return 2 * s + delta + alpha_plus + alpha_minus + mu


def threshold_2(params):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    delta = 1 - b
    s = c - 1
    return (s + alpha_plus) * (s + delta + alpha_minus + mu) - alpha_minus * alpha_plus


def q_get(params, sign):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    delta = 1 - b
    s = c - 1
    assert sign in [-1, +1]
    bterm = alpha_plus - alpha_minus - mu - delta
    return 0.5 / alpha_minus * (bterm + sign * np.sqrt(bterm ** 2 + 4 * alpha_minus * alpha_plus))


def fp_location_noflow(params):
    q1 = q_get(params, +1)
    q2 = q_get(params, -1)
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    delta = 1 - b
    s = c - 1
    conjugate_fps = [[0,0,0], [0,0,0]]
    for idx, q in enumerate([q1,q2]):
        xi = N * (s + alpha_plus - alpha_minus * q) / (s + (delta + s) * q)
        yi = q * xi
        zi = N - xi - yi
        conjugate_fps[idx] = [xi, yi, zi]
    return [[0, 0, N], conjugate_fps[0], conjugate_fps[1]]


def fp_location_sympy_system(params, system):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    sym_x = Symbol("x")
    sym_y = Symbol("y")
    VV = (v_x + v_y + v_z) / N
    if system == "feedback_z":
        z = N - sym_x - sym_y
        alpha_plus = alpha_plus * (1 + z / (z + PARAM_Z0_RATIO*N))
        alpha_minus = alpha_minus * PARAM_Z0_RATIO*N / (z + PARAM_Z0_RATIO*N)
    elif system == "feedback_yz":
        yz = N - sym_x
        alpha_plus = alpha_plus * (1 + yz / (yz + PARAM_Y0_PLUS_Z0_RATIO * N))
        alpha_minus = alpha_minus * PARAM_Y0_PLUS_Z0_RATIO * N / (yz + PARAM_Y0_PLUS_Z0_RATIO * N)
    xdot = (c-a)/N*sym_x**2 + (c-b)/N*sym_x*sym_y + (a-c-alpha_plus-VV)*sym_x + alpha_minus*sym_y + v_x
    ydot = (c-b)/N*sym_y**2 + (c-a)/N*sym_x*sym_y + (b-c-alpha_minus-mu-VV)*sym_y + alpha_plus*sym_x + v_y
    eqns = (xdot, ydot)
    solution = solve(eqns)
    solution_list = [[0,0,0], [0,0,0], [0,0,0]]
    for i in xrange(3):
        x_i = float(re(solution[i][sym_x]))
        y_i = float(re(solution[i][sym_y]))
        solution_list[i] = [x_i, y_i, N - x_i - y_i]
    return solution_list


def fp_location_sympy_quartic(params, system):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    sym_x = Symbol("x")
    sym_y = Symbol("y")
    if system == "feedback_z":
        z = N - sym_x - sym_y
        alpha_plus = alpha_plus * (1 + z / (z + PARAM_Z0_RATIO*N))
        alpha_minus = alpha_minus * PARAM_Z0_RATIO*N / (z + PARAM_Z0_RATIO*N)
    elif system == "feedback_yz":
        yz = N - sym_x
        alpha_plus = alpha_plus * (1 + yz / (yz + PARAM_Y0_PLUS_Z0_RATIO * N))
        alpha_minus = alpha_minus * PARAM_Y0_PLUS_Z0_RATIO * N / (yz + PARAM_Y0_PLUS_Z0_RATIO * N)
    VV = (v_x+v_y+v_z)/N
    a0 = (c-a)/N
    a1 = 0.0
    b0 = 0.0
    b1 = (c-b)/N
    c0 = (c-b)/N
    c1 = (c-a)/N
    d0 = (a-c-alpha_plus-VV)
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


def fsolve_func(xvec_guess, system, params):  # TODO: faster if split into 3 fns w.o if else for feedback cases
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    VV = (v_x + v_y + v_z) / N
    x0, y0 = xvec_guess
    if system == "feedback_z":
        z = N - x0 - y0
        alpha_plus = alpha_plus * (1 + z / (z + PARAM_Z0_RATIO * N))
        alpha_minus = alpha_minus * PARAM_Z0_RATIO * N / (z + PARAM_Z0_RATIO * N)
    elif system == "feedback_yz":
        yz = N - x0
        alpha_plus = alpha_plus * (1 + yz / (yz + PARAM_Y0_PLUS_Z0_RATIO * N))
        alpha_minus = alpha_minus * PARAM_Y0_PLUS_Z0_RATIO * N / (yz + PARAM_Y0_PLUS_Z0_RATIO * N)
    xdot = (c-a)/N*x0**2 + (c-b)/N*x0*y0 + (a-c-alpha_plus-VV)*x0 + alpha_minus*y0 + v_x
    ydot = (c-b)/N*y0**2 + (c-a)/N*x0*y0 + (b-c-alpha_minus-mu-VV)*y0 + alpha_plus*x0 + v_y
    return [xdot, ydot]


def fp_location_fsolve(params, system, check_near_traj_endpt=True, gridsteps=15, tol=10e-1):
    N = params[PARAMS_ID_INV["N"]]
    unique_solutions = []
    # first check for roots near trajectory endpoints (possible stable roots)
    if check_near_traj_endpt:
        traj, _, _, _ = trajectory.trajectory_simulate(params, system, flag_showplt=False, flag_saveplt=False)
        fp_guess = traj[-1][0:2]
        solution, infodict, _, _ = fsolve(fsolve_func, fp_guess, (system, params), full_output=True)
        if np.linalg.norm(infodict["fvec"]) <= 10e-3:  # only append actual roots (i.e. f(x)=0)
            unique_solutions.append([solution[0], solution[1], N - solution[0] - solution[1]])
    # grid search of solution space (positive simplex):
    for i in xrange(gridsteps):
        x_guess = N*i/float(gridsteps)
        for j in xrange(gridsteps-i):
            y_guess = N * i / float(gridsteps)
            # TODO: this returns jacobian estimate.. use it
            solution, infodict, _, _ = fsolve(fsolve_func, [x_guess, y_guess], (system, params), full_output=True)
            append_flag = True
            for k, usol in enumerate(unique_solutions):
                if np.abs(solution[0] - usol[0]) <= tol:   # only store unique roots from list of all roots
                    append_flag = False
                    break
            if append_flag:
                if np.linalg.norm(infodict["fvec"]) <= 10e-3:    # only append actual roots (i.e. f(x)=0)
                    unique_solutions.append([solution[0], solution[1], N - solution[0] - solution[1]])
    return unique_solutions


def fp_location_general(params, system, solver_fsolve=True, solver_fast=False, solver_explicit=False):
    # TODO: sympy solver often fails when feedback added in
    # TODO: cleanup the flags here
    assert system in ODE_SYSTEMS
    if solver_fsolve:
        return fp_location_fsolve(params, system)
    elif solver_explicit:
        assert system == "default"
        return fp_location_noflow(params)
    else:
        if solver_fast:
            return fp_location_sympy_quartic(params, system)
        else:
            return fp_location_sympy_system(params, system)


def jacobian_3d(params, fp):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    M = np.array([[a - alpha_plus, alpha_minus, 0],
                  [alpha_plus, b - alpha_minus - mu, 0],
                  [0, mu, c]])
    x, y, z = fp
    diag = a*x + b*y + c*z + v_x + v_y + v_z
    r1 = [diag + x*a, x*b, x*c]
    r2 = [y*a, diag + y*b, y*c]
    r3 = [z*a, z*b, diag + z*c]
    return M - 1/N*np.array([r1,r2,r3])


def jacobian_numerical_2d(params, fp, system):
    # TODO: can use numdifftools jacobian function instead
    # TODO: move scope of func xdot etc up and use them both in func fsolve
    def func_xdot(fp):
        x, y = fp[0], fp[1]
        alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
        VV = (v_x + v_y + v_z) / N
        if system == "feedback_z":
            z = N - x - y
            alpha_plus = alpha_plus * (1 + z / (z + PARAM_Z0_RATIO * N))
            alpha_minus = alpha_minus * PARAM_Z0_RATIO * N / (z + PARAM_Z0_RATIO * N)
        elif system == "feedback_yz":
            yz = N - x
            alpha_plus = alpha_plus * (1 + yz / (yz + PARAM_Y0_PLUS_Z0_RATIO * N))
            alpha_minus = alpha_minus * PARAM_Y0_PLUS_Z0_RATIO * N / (yz + PARAM_Y0_PLUS_Z0_RATIO * N)
        return (c - a) / N * x ** 2 + (c - b) / N * x * y + (a - c - alpha_plus - VV) * x + alpha_minus * y + v_x
    def func_ydot(fp):
        x, y = fp[0], fp[1]
        alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
        VV = (v_x + v_y + v_z) / N
        if system == "feedback_z":
            z = N - x - y
            alpha_plus = alpha_plus * (1 + z / (z + PARAM_Z0_RATIO * N))
            alpha_minus = alpha_minus * PARAM_Z0_RATIO * N / (z + PARAM_Z0_RATIO * N)
        elif system == "feedback_yz":
            yz = N - x
            alpha_plus = alpha_plus * (1 + yz / (yz + PARAM_Y0_PLUS_Z0_RATIO * N))
            alpha_minus = alpha_minus * PARAM_Y0_PLUS_Z0_RATIO * N / (yz + PARAM_Y0_PLUS_Z0_RATIO * N)
        return (c-b)/N*y**2 + (c-a)/N*x*y + (b-c-alpha_minus-mu-VV)*y + alpha_plus*x + v_y
    epsilon = 10e-4
    row_x = approx_fprime(fp, func_xdot, epsilon)
    row_y = approx_fprime(fp, func_ydot, epsilon)
    return np.array([row_x, row_y])


def is_stable(params, fp, system, method="numeric_2d"):
    if method == "numeric_2d":
        assert len(fp) == 2
        J = jacobian_numerical_2d(params, fp, system)
        eigenvalues, V = np.linalg.eig(J)
    elif method == "algebraic_3d":
        J = jacobian_3d(params, fp)
        eigenvalues, V = np.linalg.eig(J)
    else:
        raise ValueError("method must be 'numeric_2d' or 'algebraic_3d'")
    return all(eig < 0 for eig in eigenvalues)


def get_physical_and_stable_fp(params, ode_system):
    fp_locs = fp_location_general(params, ode_system, solver_fsolve=True)
    fp_locs_physical_and_stable = []
    for fp in fp_locs:
        if all([val > -0.1 for val in fp]):
            if is_stable(params, fp[0:2], ode_system, method="numeric_2d"):
                fp_locs_physical_and_stable.append(fp)
                #eigs,V = np.linalg.eig(jacobian_numerical_2d(params, fp[0:2], ode_system))
                #print fp, eigs
    return fp_locs_physical_and_stable
