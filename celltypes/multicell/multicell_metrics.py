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


def calc_compression_ratio(x, eta_0=None, datatype='full', elemtype=np.bool, method='manual'):
    """
    TODO compare vs conv 01 -1 1 on state array
    x - the data object (assume lies between -1 and 1)
    eta_0 is meant to be a rough upper bound on eta(x)
        - compute via 'maximally disordered' input x (random data)
    Returns eta(x)/eta_0 ranging between 0 and 1 +- eps
    """
    assert method in ['manual', 'package']
    assert datatype in ['full', 'custom']
    assert elemtype in [np.bool, np.int, np.float]

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
        if eta_0 is None:
            assert -1 <= np.min(x) <= np.max(x) <= 1
            #x_random = np.random.rand(*(x.shape))*2 - 1  # TODO flag ref as float or bool
            if elemtype==np.bool:
                if x.dtype!=np.bool:
                    print 'NOTE: Recasting x as np.float from', x.dtype
                    x = x.astype(dtype=np.bool)
                x_random = np.random.randint(0, high=2, size=x.shape, dtype=np.bool)
            elif elemtype==np.int:
                assert np.issubdtype(x.dtype, np.int)
                nint = len(set(x))
                x_random = np.random.randint(0, high=nint+1, size=x.shape, dtype=np.int)
            else:
                assert elemtype==np.float
                if x.dtype!=np.float:
                    print 'NOTE: Recasting x as np.float from', x.dtype
                    x = x.astype(dtype=np.float)
                x_random = np.random.rand(*(x.shape)) * 2 - 1
            eta_0 = foo(x_random)  # consider max over few realizations?
        eta = foo(x)
    return float(eta)/eta_0, eta, eta_0


def test_compression_ratio():
    nn = 10000
    print "test_compression_ratio for x: %dx1 array..." % nn
    x1 = np.ones(nn, dtype=np.int)  #[1, 1, 1, 1]
    x2 = np.zeros(nn, dtype=np.int) #[-1, -1, -1, -1]
    x3 = np.random.randint(0, high=2, size=nn)
    eta_ratio_1, eta_1, eta_0_1 = calc_compression_ratio(x1, eta_0=None, datatype='custom', method='manual', elemtype=np.bool)
    eta_ratio_2, eta_2, eta_0_2 = calc_compression_ratio(x2, eta_0=None, datatype='custom', method='manual', elemtype=np.bool)
    eta_ratio_3, eta_3, eta_0_3 = calc_compression_ratio(x3, eta_0=None, datatype='custom', method='manual', elemtype=np.bool)
    print 'x1', 'gives', eta_ratio_1, eta_1, eta_0_1
    print 'x2', 'gives', eta_ratio_2, eta_2, eta_0_2
    print 'x3', 'gives', eta_ratio_3, eta_3, eta_0_3

    xshape = (1000, 500)
    print "test_compression_ratio for x: %d x %d array..." % (xshape[0], xshape[1])
    x1 = np.ones(xshape)
    x2 = -np.ones(xshape)
    x3 = np.zeros(xshape)
    x4 = np.zeros(xshape)
    x4[:,0] = 1
    x5 = np.random.rand(*xshape)*2 - 1
    print x5.shape
    x6 = np.random.randint(-1, high=2, size=xshape)
    x7 = np.random.randint(0, high=2, size=xshape) * 2 - 1
    x_dict ={1: {'data': x1, 'label': 'all +1', 'dtype': np.bool},
             2: {'data': x2, 'label': 'all -1', 'dtype': np.bool},
             3: {'data': x3, 'label': 'all 0',  'dtype': np.bool},
             4: {'data': x4, 'label': 'all 0 except all 1 first col', 'dtype': np.bool},
             5: {'data': x5, 'label': 'rand floats -1 to 1', 'dtype': np.float},
             6: {'data': x6, 'label': 'rand ints -1, 0, 1', 'dtype': np.float},
             7: {'data': x7, 'label': 'rand ints -1, 1', 'dtype': np.float}}
    for idx in xrange(1, len(x_dict.keys())+1):
        elem = x_dict[idx]['data']
        elemtype = x_dict[idx]['dtype']
        eta_ratio, eta, eta_0 = calc_compression_ratio(elem, eta_0=None, datatype='custom', method='manual',
                                                       elemtype=elemtype)
        print x_dict[idx]['label'], 'gives', eta_ratio, eta, eta_0
    return None


if __name__ == '__main__':
    test_compression_ratio()
