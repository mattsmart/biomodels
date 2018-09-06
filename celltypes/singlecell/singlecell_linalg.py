import numpy as np

from singlecell_constants import J_RANDOM_DELETE_RATIO


def memory_corr_matrix_and_inv(xi, check_invertible=False):
    if check_invertible:
        print xi.shape, np.linalg.matrix_rank(xi)  # expect rank = p (num memories) for invertibility
    corr_matrix = np.dot(xi.T, xi) / len(xi)
    return corr_matrix, np.linalg.inv(corr_matrix)


def interaction_matrix(xi, corr_inv, method, flag_prune_intxn_matrix=False):
    print "Note network method for interaction_matrix() is %s" % method
    if method == "hopfield":
        intxn_matrix = np.dot(xi, xi.T) / len(xi)
    elif method == "projection":
        intxn_matrix = reduce(np.dot, [xi, corr_inv, xi.T]) / len(xi)
    else:
        raise ValueError("method arg invalid, must be one of %s" % ["projection", "hopfield"])
    np.fill_diagonal(intxn_matrix, 0)
    if flag_prune_intxn_matrix:
        randarr = np.random.rand(*intxn_matrix.shape)
        randarr = np.where(randarr > J_RANDOM_DELETE_RATIO, 1, 0)
        intxn_matrix = intxn_matrix * randarr
    return intxn_matrix


def predictivity_matrix(xi, corr_inv):
    return np.dot(corr_inv, xi.T) / len(xi)  # eta_ij is the "predictivity" of TF i in cell fate j
