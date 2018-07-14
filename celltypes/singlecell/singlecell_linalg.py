import numpy as np

from singlecell_constants import J_RANDOM_DELETE_RATIO


def memory_corr_matrix_and_inv(xi, check_invertible=False):

    if check_invertible:
        print xi.shape, np.linalg.matrix_rank(xi)  # expect rank = p (num memories) for invertibility

    corr_matrix = np.dot(xi.T, xi) / len(xi)
    return corr_matrix, np.linalg.inv(corr_matrix)


def interaction_matrix(xi, corr_inv, method, flag_prune_intxn_matrix=False):
    if method == "hopfield":
        j = np.dot(xi, xi.T) / len(xi[0])                         # TODO: not sure if factor 1/N or 1/p needed...
    elif method == "projection":
        j = reduce(np.dot, [xi, corr_inv, xi.T]) / len(xi)     # TODO: not sure if factor 1/N needed
    else:
        raise ValueError("method arg invalid, must be one of %s" % ["projection", "hopfield"])
    np.fill_diagonal(j, 0)                                    # TODO: is this step necessary in both cases? speedup...
    if flag_prune_intxn_matrix:
        randarr = np.random.rand(len(j), len(j[0]))
        randarr = np.where(randarr > J_RANDOM_DELETE_RATIO, 1, 0)
        #print randarr
        j = j * randarr
    return j


def predictivity_matrix(xi, corr_inv):
    return np.dot(corr_inv, xi.T) / len(xi)  # eta_ij is the "predictivity" of TF i in cell fate j
