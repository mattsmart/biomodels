import numpy as np

from singlecell_constants import J_RANDOM_DELETE_RATIO, HOLLOW_INTXN_MATRIX


def memory_corr_matrix_and_inv(xi, check_invertible=False):
    if check_invertible:
        print xi.shape, np.linalg.matrix_rank(xi)  # expect rank = p (num memories) for invertibility
    corr_matrix = np.dot(xi.T, xi) / float(xi.shape[0])
    return corr_matrix, np.linalg.inv(corr_matrix)


def interaction_matrix(xi, corr_inv, method, flag_prune_intxn_matrix=False, hollow=HOLLOW_INTXN_MATRIX):
    print "Note network method for interaction_matrix() is %s" % method
    if method == "hopfield":
        intxn_matrix = np.dot(xi, xi.T) / float(xi.shape[0])
    elif method == "projection":
        intxn_matrix = reduce(np.dot, [xi, corr_inv, xi.T]) / float(xi.shape[0])
    else:
        raise ValueError("method arg invalid, must be one of %s" % ["projection", "hopfield"])
    if not hollow:
        print("Warning, hollow flag set to False in single-cell constants (for intxn matrix build)")
    else:
        np.fill_diagonal(intxn_matrix, 0)
    if flag_prune_intxn_matrix:
        randarr = np.random.rand(*intxn_matrix.shape)
        randarr = np.where(randarr > J_RANDOM_DELETE_RATIO, 1, 0)
        intxn_matrix = intxn_matrix * randarr
    return intxn_matrix


def predictivity_matrix(xi, corr_inv):
    return np.dot(corr_inv, xi.T) / float(xi.shape[0])  # eta_ij is the "predictivity" of TF i in cell fate j


def sorted_eig(arr, take_real=True):
    # TODO care with the real, assert matrix symmetric for now?
    E_unsorted, V_unsorted = np.linalg.eig(arr)
    if take_real:
        E_unsorted = np.real(E_unsorted)
        V_unsorted = np.real(V_unsorted)
    sortlist = np.argsort(E_unsorted)
    evals = E_unsorted[sortlist]
    evecs = V_unsorted[:, sortlist]
    return evals, evecs
