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
from sympy import Symbol, solve

from constants import PARAMS_ID


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


def fp_location_numeric(params):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    sym_x = Symbol("x")
    sym_y = Symbol("y")
    sym_z = Symbol("z")
    xdot = (c-a)/N*sym_x**2 + (c-b)/N*sym_x*sym_y + (a-c-alpha_plus-(v_x+v_y+v_z)/N)*sym_x + alpha_minus*sym_y + v_x
    ydot = (c-b)/N*sym_y**2 + (c-a)/N*sym_x*sym_y + (b-c-alpha_minus-mu-(v_x+v_y+v_z)/N)*sym_y + alpha_plus*sym_x + v_y
    pop_constraint = N - sym_x - sym_y - sym_z
    eqns = (xdot, ydot, pop_constraint)
    solution = solve(eqns)
    orderdict = {0: sym_x, 1: sym_y, 2: sym_z}
    sol_a = [float(solution[0][orderdict[i]]) for i in xrange(3)]
    sol_b = [float(solution[1][orderdict[i]]) for i in xrange(3)]
    sol_c = [float(solution[2][orderdict[i]]) for i in xrange(3)]
    return [sol_a, sol_b, sol_c]


def fp_location_general(params):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    if v_x == 0 and v_y == 0 and v_z == 0:
        return fp_location_noflow(params)
    else:
        return fp_location_numeric(params)


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
    all(eig < 0 for eig in eigenvalues)


def write_bifurc_data(bifurcation_search, x0_array, x1_array, x2_array, bifurc_id, filedir, filename):
    filepath = filedir + sep + filename
    with open(filepath, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        csv_header = [bifurc_id, 'x0_x', 'x0_y', 'x0_z', 'x1_x', 'x1_y', 'x1_z', 'x2_x', 'x2_y', 'x2_z']
        writer.writerow(csv_header)
        for idx in xrange(len(bifurcation_search)):
            line = [bifurcation_search[idx]] + list(x0_array[idx,:]) + list(x1_array[idx,:]) + list(x2_array[idx,:])
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
