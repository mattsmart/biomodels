import numpy as np


def step_state_load(fpath, cells_as_cols=True, num_genes=None, num_cells=None):
    """
    Loads the data from a state save file which is int text file
        num_genes x num_cells, or
        total_spins
    If cells_as_cols, then make sure loaded data is already 2D arr and return that
    else,
    """
    X = np.loadtxt(fpath)
    ndim = len(X.shape)
    assert ndim in [1,2]
    if ndim == 1:
        if cells_as_cols:
            X = X.reshape((num_genes, num_cells), order='F')  # reshape as 2D arr
    else:
        if not cells_as_cols:
            X = X.reshape((num_genes * num_cells), order='F')  # reshape as 1D arr
    return X

