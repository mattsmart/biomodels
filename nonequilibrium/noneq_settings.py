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
        J = 0

    return J
