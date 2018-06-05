import numpy as np
import unittest

from expt_analysis import build_basin_states
from singlecell_functions import hamiltonian, hamming


class TestExpt(unittest.TestCase):

    def test_something(self):
        print "RUNNING test_something"
        self.assertEqual(True, False)

    def test_build_basin_states(self):
        print "RUNNING test_build_basin_states"

        intxn_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        memory_vec = np.array([1, 1, 1])
        flip1 = np.array([-1, 1, 1])
        flip2 = np.array([1, -1, 1])
        flip3 = np.array([1, 1, -1])
        flip12 = np.array([-1, -1, 1])
        flip23 = np.array([1, -1, -1])
        flip31 = np.array([-1, 1, -1])
        flip123 = np.array([-1, -1, -1])

        states = [memory_vec, flip1, flip2, flip3, flip12, flip23, flip31, flip123]
        for state in states:
            print state
            print hamming(memory_vec, state)
            print hamiltonian(state, intxn_matrix=intxn_matrix)

        basin = build_basin_states(intxn_matrix, memory_vec)
        print basin

        self.assertIn(memory_vec, basin)
        self.assertIn(flip1, basin)
        self.assertIn(flip2, basin)
        self.assertIn(flip3, basin)
        self.assertNotIn(flip12, basin)
        self.assertNotIn(flip23, basin)
        self.assertNotIn(flip31, basin)
        self.assertNotIn(flip123, basin)



if __name__ == '__main__':
    unittest.main()
