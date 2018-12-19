import numpy as np


def vectorize_matrix(arr, order='C'):
    """
    Convert NxN matrix into N^2 1-dim array, row by row
        C - C-style, do row-by-row
        F - Fortran-style, do column-by-column
    """
    assert all([len(arr.shape) == 2, arr.shape[0] == arr.shape[1], order in ['C', 'F']])
    vec = arr.flatten(order=order)
    return vec

def tile_with_identity(arr):
    """
    Computes the block outer produce "A cross I" which will be of size N^2 x N^2
    """
    assert len(arr.shape) == 2 and arr.shape[0] == arr.shape[1]
    N = arr.shape[0]
    #TODO
    return None


