import numpy as np
import os

# IO
RUNS_FOLDER = "runs" + os.sep             # store timestamped runs here


# SIMULATION CONSTANTS
BETA = 3.0
NUM_STEPS = 100
DEFAULT_N = 3


# SIMULATION SETTINGS (J constructions given N)
def build_J(N, id='symm', asymm_scale=0.01):

    if N == 3:
        J_symm = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])
        J_broken1 = np.array([[0, 1, 2],
                              [0.5, 0, 1],
                              [1, 1, 0]])
        J_broken2 = np.array([[0, 1, 2],
                              [-0.5, 0, 1],
                              [1, 1, 0]])
        J_broken3 = np.array([[0, 0.1, 6],
                              [-1, 0, -0.1],
                              [-4, 10, 0]])
        J_general = np.array([[0, -61, -100],
                              [-9, 0, -1],
                              [87, 11, 0]])
        if id == 'symm':
            J = J_symm
        elif id == 'asymm_1':
            J = J_broken1
        elif id == 'asymm_2':
            J = J_broken2
        elif id == 'asymm_3':
            J = J_broken3
        else:
            J = J_general

    elif N == 4:
        if id == 'symm':
            asymm_scale = 0.0
        mem = [[1 for i in xrange(N)]]
        XI = np.transpose(np.array(mem))
        J = np.dot(XI, np.transpose(XI))
        J = J + asymm_scale * np.random.uniform(-5.0, 5.0, (N, N))
        np.fill_diagonal(J, 0)

    else:
        print "warning, N=%d not implemented" % N
        J = np.random.uniform(-1,1,(N,N))

    return J


# VISUALIZATION SETTINGS:
def get_network_pos(N):
    if N==3:
        pos = {(1, 1, 0): np.array([0.83582596, 0.47751457]), (0, 1, 1): np.array([0.05537278, 0.]),
               (1, 0, 0): np.array([0.78376018, 1.]), (0, 0, 1): np.array([0., 0.52530576]),
               (1, 0, 1): np.array([0.50786198, 0.62561025]), (0, 0, 0): np.array([0.26718072, 0.89391983]),
               (0, 1, 0): np.array([0.32738261, 0.37897392]), (1, 1, 1): np.array([0.56842864, 0.11116961])}
    else:
        return None