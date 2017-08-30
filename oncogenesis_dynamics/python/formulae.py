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
from scipy.integrate import ode, odeint
from sympy import Symbol, solve, re

from constants import PARAMS_ID, CSV_DATA_TYPES, ODE_METHODS, PARAM_Z0_RATIO, PARAM_HILL


def system_vector(init_cond, times, alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z):
    x, y, z = init_cond
    fbar = (a * x + b * y + c * z + v_x + v_y + v_z) / N
    dxdt = v_x - x * alpha_plus + y * alpha_minus + (a - fbar) * x
    dydt = v_y + x * alpha_plus - y * (alpha_minus + mu) + (b - fbar) * y
    dzdt = v_z + y * mu + (c - fbar) * z
    return [dxdt, dydt, dzdt]


def system_vector_obj_ode(t_scalar, r_idx, alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z):
    return system_vector(r_idx, t_scalar, alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z)


def system_vector_feedback(init_cond, times, alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z):
    x, y, z = init_cond
    alpha_plus = alpha_plus * z / (z + PARAM_Z0_RATIO*N)
    alpha_minus = alpha_minus * 1 / (z + PARAM_Z0_RATIO*N)
    #alpha_plus = alpha_plus * 1 / (z + PARAM_Z0_RATIO * N)
    #alpha_minus = alpha_minus * z / (z + PARAM_Z0_RATIO * N)
    fbar = (a * x + b * y + c * z + v_x + v_y + v_z) / N
    dxdt = v_x - x * alpha_plus + y * alpha_minus + (a - fbar) * x
    dydt = v_y + x * alpha_plus - y * (alpha_minus + mu) + (b - fbar) * y
    dzdt = v_z + y * mu + (c - fbar) * z
    return [dxdt, dydt, dzdt]


def system_vector_obj_ode_feedback(t_scalar, r_idx, alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z):
    return system_vector_feedback(r_idx, t_scalar, alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z)


def ode_euler(init_cond, times, params, system):
    if system == "default":
        fn = system_vector
    else:
        fn = system_vector_feedback
    dt = times[1] - times[0]
    r = np.zeros((len(times), 3))
    r[0] = np.array(init_cond)
    for idx, t in enumerate(times[:-1]):
        v = fn(r[idx], None, *params)
        r[idx+1] = r[idx] + np.array(v)*dt
    return r


def ode_rk4(init_cond, times, params, system):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    dt = times[1] - times[0]
    r = np.zeros((len(times), 3))
    r[0] = np.array(init_cond)
    """
    def system_vector_obj_ode(t_scalar, r_idx, alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z):
        x, y, z = r_idx
        fbar = (a * x + b * y + c * z + v_x + v_y + v_z) / N
        dxdt = v_x - x * alpha_plus + y * alpha_minus + (a - fbar) * x
        dydt = v_y + x * alpha_plus - y * (alpha_minus + mu) + (b - fbar) * y
        dzdt = v_z + y * mu + (c - fbar) * z
        return [dxdt, dydt, dzdt]
    def system_vector_obj_ode_feedback(t_scalar, r_idx, alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z):
        x, y, z = r_idx
        NEWPARAM_z0 = 1
        alpha_plus = alpha_plus * z / (z + NEWPARAM_z0)
        alpha_minus = alpha_minus * 1 / (z + NEWPARAM_z0)
        fbar = (a * x + b * y + c * z + v_x + v_y + v_z) / N
        dxdt = v_x - x * alpha_plus + y * alpha_minus + (a - fbar) * x
        dydt = v_y + x * alpha_plus - y * (alpha_minus + mu) + (b - fbar) * y
        dzdt = v_z + y * mu + (c - fbar) * z
        return [dxdt, dydt, dzdt]
    if system == "default":
        fn = system_vector_obj_ode
    else:
        fn = system_vector_obj_ode_feedback
    """
    if system == "default":
        fn = system_vector_obj_ode
    else:
        fn = system_vector_obj_ode_feedback
    obj_ode = ode(fn, jac=None)
    obj_ode.set_initial_value(init_cond, times[0])
    obj_ode.set_f_params(*params)
    obj_ode.set_integrator('dopri5')
    idx = 1
    while obj_ode.successful() and obj_ode.t < times[-1]:
        obj_ode.integrate(obj_ode.t + dt)
        r[idx] = np.array(obj_ode.y)
        idx += 1
    return r


def ode_libcall(init_cond, times, params, system):
    if system == "default":
        fn = system_vector
    else:
        fn = system_vector_feedback
    r = odeint(fn, init_cond, times, args=tuple(params))
    return r


def ode_general(init_cond, times, params, method="libcall", system="default"):
    assert system in ["default", "feedback"]
    if method == "libcall":
        return ode_libcall(init_cond, times, params, system)
    elif method == "rk4":
        return ode_rk4(init_cond, times, params, system)
    elif method == "euler":
        return ode_euler(init_cond, times, params, system)
    else:
        raise ValueError("method arg invalid, must be one of %s" % ODE_METHODS)


def fp_from_timeseries(r, tol=0.001):
    fp_test = r[-1,:]
    fp_check = r[-2,:]
    if np.linalg.norm(fp_test - fp_check) <= tol:
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
    # assumes threshold_2 is stronger constraint, atm hardcode rearrange expression for bifruc param
    if bifurc_name == "bifurc_b":
        delta_val = alpha_minus * alpha_plus / (s + alpha_plus) - (s + alpha_minus + mu)
        bifurc_val = 1 - delta_val
        return bifurc_val
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


def fp_location_numeric_system(params):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    sym_x = Symbol("x")
    sym_y = Symbol("y")

    VV = (v_x+v_y+v_z)/N
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


def fp_location_numeric_quartic(params):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    sym_x = Symbol("x")
    sym_y = Symbol("y")

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


def fp_location_general(params, solver_numeric=True, solver_fast=True):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    if solver_numeric:
        if solver_fast:
            return fp_location_numeric_quartic(params)
        else:
            return fp_location_numeric_system(params)
    else:
        return fp_location_noflow(params)


def jacobian3d(params, fp):
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


def is_stable(params, fp):
    J = jacobian3d(params, fp)
    eigenvalues, V = np.linalg.eig(J)
    return all(eig < 0 for eig in eigenvalues)


def write_bifurc_data(bifurcation_search, x0, x0_stab, x1, x1_stab, x2, x2_stab, bifurc_id, filedir, filename):
    csv_header = [bifurc_id, 'x0_x', 'x0_y', 'x0_z', 'x0_stab', 'x1_x', 'x1_y', 'x1_z', 'x1_stab', 'x2_x', 'x2_y',
                  'x2_z', 'x2_stab']
    filepath = filedir + sep + filename
    with open(filepath, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(csv_header)
        for idx in xrange(len(bifurcation_search)):
            line = [bifurcation_search[idx]] + list(x0[idx,:]) + list(x0_stab[idx]) + list(x1[idx,:]) + \
                   list(x1_stab[idx]) + list(x2[idx,:]) + list(x2_stab[idx])
            writer.writerow(line)
    return filepath


def write_params(params, filedir, filename):
    filepath = filedir + sep + filename
    with open(filepath, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for idx in xrange(len(PARAMS_ID)):
            if params[idx] is None:
                params[idx] = 'None'
            writer.writerow([PARAMS_ID[idx], params[idx]])
    return filepath


def read_bifurc_data(filedir, filename):
    def str_to_data(elem):
        if elem == 'True':
            return True
        elif elem == 'False':
            return False
        else:
            return elem
    with open(filedir + sep + filename, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        nn = sum(1 for row in datareader) - 1
        csvfile.seek(0)
        header = datareader.next()
        data_dict = {key: np.zeros((nn, 1), dtype=CSV_DATA_TYPES[key]) for key in header}
        for idx_row, row in enumerate(datareader):
            for idx_col, elem in enumerate(row):
                data_dict[header[idx_col]][idx_row] = str_to_data(elem)
    return data_dict


def read_params(filedir, filename):
    with open(filedir + sep + filename, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        num_params = sum(1 for row in datareader)
        csvfile.seek(0)
        params = [0.0]*num_params
        for idx, pair in enumerate(datareader):
            assert pair[0] == PARAMS_ID[idx]
            if pair[1] != 'None':
                params[idx] = float(pair[1])
            else:
                params[idx] = None
    return params
