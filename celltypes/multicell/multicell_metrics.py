import numpy as np
import os

from singlecell.singlecell_functions import hamiltonian

def calc_lattice_energy(lattice, simsetup, field, fs, gamma, search_radius, ratio_to_remove, exosome_string):
    """
    Lattice energy is the multicell hamiltonian
        H_multi = Sum (H_i) + gamma * Sum (interactions)
    Returns total energy and the two main terms
    """
    M1 = len(lattice)
    M2 = len(lattice[0])
    num_cells = M1 * M2
    assert M1 == M2  # TODO relax
    H_multi = 0
    term1 = 0
    term2 = 0
    # compute self energies
    for i in xrange(M1):
        for j in xrange(M2):
            cell = lattice[i][j]
            term1 += hamiltonian(cell.get_current_state(), simsetup['J'], field=field, fs=fs)
    # compute interactions  # TODO check validity
    for i in xrange(M1):
        for j in xrange(M2):
            cell = lattice[i][j]
            nbr_states_sent, neighbours = cell.get_local_exosome_field(lattice, search_radius, M1,
                                                                       exosome_string=exosome_string,
                                                                       ratio_to_remove=ratio_to_remove)
            if simsetup['FIELD_SEND'] is not None:
                nbr_states_sent += cell.get_local_paracrine_field(lattice, neighbours, simsetup)
            nbr_states_sent_01 = (nbr_states_sent + len(neighbours)) / 2.0
            field_neighbours = np.dot(simsetup['FIELD_SEND'], nbr_states_sent_01)
            term2 += np.dot(field_neighbours, cell.get_current_state())
    term2_scaled = term2 * gamma / 2  # divide by two because of double-counting neighbours
    H_multi = term1 - term2_scaled
    return H_multi, term1, term2_scaled


def calc_compression_ratio(x, eta_0=None, datatype='full', method='manual'):
    """
    TODO dos
    x - the data object (assume lies between -1 and 1)
    eta_0 is meant to be a rough upper bound on eta(x)
        - compute via 'maximally disordered' input x (random data)
    Returns eta(x)/eta_0 ranging between 0 and 1 +- eps
    """
    assert method in ['manual', 'package']
    assert datatype in ['full', 'custom']

    def foo(x_to_compress):
        fname = 'tmp.npz'
        np.savez_compressed(fname, a=x_to_compress)
        fsize = os.path.getsize(fname)
        os.remove(fname)
        return fsize

    if datatype == 'full':
        if eta_0 is None:
            # TODO compute eta_0
            x_random = None
            eta_0 = foo(x_random)  # consider max over few realizations?
        eta = foo(x)
    else:
        assert datatype == 'custom'
        x = np.array(x)
        if eta_0 is None:
            assert -1 <= np.min(x) <= np.max(x) <= 1
            x_random = np.random.rand(*(x.shape))*2 - 1
            eta_0 = foo(x_random)  # consider max over few realizations?
        eta = foo(x)
    return float(eta)/eta_0, eta, eta_0


def test_compression_ratio():
    print "test_compression_ratio for x: 4x1 list..."
    x1 = [1, 1, 1, 1]
    x2 = [-1, -1, -1, -1]
    x3 = [1, -1, 1, -1]
    eta_ratio_1, eta_1, eta_0_1 = calc_compression_ratio(x1, eta_0=None, datatype='custom', method='manual')
    eta_ratio_2, eta_2, eta_0_2 = calc_compression_ratio(x2, eta_0=None, datatype='custom', method='manual')
    eta_ratio_3, eta_3, eta_0_3 = calc_compression_ratio(x3, eta_0=None, datatype='custom', method='manual')
    print x1, 'gives', eta_ratio_1, eta_1, eta_0_1
    print x2, 'gives', eta_ratio_2, eta_2, eta_0_2
    print x3, 'gives', eta_ratio_3, eta_3, eta_0_3
    print "test_compression_ratio for x: 100x50 array..."
    xshape = (100, 50)
    x1 = np.ones(xshape)
    x2 = -np.ones(xshape)
    x3 = np.zeros(xshape)
    x4 = np.zeros(xshape)
    x4[:,0] = 1
    x5 = np.random.rand(*xshape)*2 - 1
    x6 = np.random.randint(-1, high=1, size=xshape)
    x7 = np.random.randint(0, high=1, size=xshape) * 2 - 1
    x_list = [x1, x2, x3, x4, x5, x6, x7]
    x_labels = ['all +1', 'all -1', 'all 0',
                'all 0 except all 1 first col', 'rand floats -1 to 1',
                'rand ints -1, 0, 1', 'rand ints -1, 1']
    for idx, elem in enumerate(x_list):
        eta_ratio, eta, eta_0 = calc_compression_ratio(elem, eta_0=None, datatype='custom', method='manual')
        print x_labels[idx], 'gives', eta_ratio, eta, eta_0
    return None


if __name__ == '__main__':
    test_compression_ratio()
