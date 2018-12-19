import numpy as np


def vectorize_matrix(arr, order='C'):
    """
    Convert NxN matrix into N^2 1-dim array, row by row
        C - C-style, do row-by-row (default)
        F - Fortran-style, do column-by-column
    """
    assert all([len(arr.shape) == 2, arr.shape[0] == arr.shape[1], order in ['C', 'F']])
    vec = arr.flatten(order=order)
    return vec


def arr_cross_eye(arr):
    """
    Computes the block outer produce "A cross I" which will be of size N^2 x N^2
    """
    assert len(arr.shape) == 2 and arr.shape[0] == arr.shape[1]
    N = arr.shape[0]
    tiled = np.zeros((N**2, N**2))
    for i in xrange(N):
        for j in xrange(N):
            if i == j:
                idx_start = N * i
                idx_end = N * (i+1)
                tiled[idx_start:idx_end, idx_start:idx_end] = arr
    return tiled


def eye_cross_arr(arr):
    """
    Computes the block outer produce "I cross arr" which will be of size N^2 x N^2
    """
    assert len(arr.shape) == 2 and arr.shape[0] == arr.shape[1]
    N = arr.shape[0]
    tiled = np.zeros((N**2, N**2))
    # TODO
    """
    for i in xrange(N):
        for j in xrange(N):
            if i == j:
                idx_start = N * i
                idx_end = N * (i+1)
                tiled[idx_start:idx_end, idx_start:idx_end] = arr
    """
    return tiled


if __name__ == '__main__':
    print 'testing Ax=b construction'
    C_test = np.array([[1.0 , 2.0], [3.0, 4.0]])
    print 'arr\n', C_test
    arr_cross_eye = arr_cross_eye(C_test)
    eye_cross_arr = eye_cross_arr(C_test)
    print 'arr_cross_eye\n', arr_cross_eye
    print 'eye_cross_arr\n', eye_cross_arr
