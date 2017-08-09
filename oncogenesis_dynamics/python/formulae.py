"""
Conventions
- params is 7-vector of the form: params[0] -> alpha_plus
                                  params[1] -> alpha_minus
                                  params[2] -> mu
                                  params[3] -> a           (usually normalized to 1)
                                  params[4] -> b           (b = 1 - delta)
                                  params[5] -> c           (c = 1 + s)
                                  params[6] -> N
                                  params[7] -> v_x
                                  params[8] -> v_y         (typically 0)
                                  params[9] -> v_z         (typically 0)
- if an element of params is specified as None then a bifurcation range will be be found and used
"""

import csv
import numpy as np
from os import sep

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


def fp_location(params, q):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    delta = 1 - b
    s = c - 1
    xi = N * (s + alpha_plus - alpha_minus * q) / (s + (delta + s) * q)
    yi = q * xi
    zi = N - xi - yi
    return xi, yi, zi


def write_bifurc_data(bifurcation_search, x1_array, x2_array, bifurc_id, filedir, filename):
    filepath = filedir + sep + filename
    with open(filepath, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        csv_header = [bifurc_id, 'x1_x', 'x1_y', 'x1_z', 'x2_x', 'x2_y', 'x2_z']
        writer.writerow(csv_header)
        for idx in xrange(len(bifurcation_search)):
            line = [bifurcation_search[idx]] + list(x1_array[idx,:]) + list(x2_array[idx,:])
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
