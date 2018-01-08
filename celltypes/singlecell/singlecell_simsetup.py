import numpy as np
import re

from singlecell_constants import METHOD, FLAG_BOOL, FLAG_REMOVE_DUPES, ZSCORE_DATAFILE_PATH, J_RANDOM_DELETE_RATIO, FLAG_PRUNE_INTXN_MATRIX

"""
Conventions follow from Lang & Mehta 2014, PLOS Comp. Bio
- note the memory matrix is transposed throughout here (dim N x p instead of dim p x N)
"""


def load_singlecell_data(zscore_datafile=ZSCORE_DATAFILE_PATH):
    """
    Returns list of cell types (size p), list of TFs (size N), and xi array where xi_ij is ith TF value in cell type j
    Note the Mehta SI file has odd formatting (use regex to parse); array text file is read in as single line:
    http://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1003734.s005&type=supplementary
    """
    gene_labels = []
    with open(zscore_datafile) as f:
        origline = f.readline()
        filelines = origline.split('\r')
        for idx_row, row in enumerate(filelines):
            row_split = re.split(r'\t', row)
            if idx_row == 0:  # celltypes line, first row
                celltype_labels = row_split[1:]
            else:
                gene_labels.append(row_split[0])
    # reloop to get data without excessive append calls
    expression_data = np.zeros((len(gene_labels), len(celltype_labels)))
    with open(zscore_datafile) as f:
        origline = f.readline()
        filelines_dataonly = origline.split('\r')[1:]
        for idx_row, row in enumerate(filelines_dataonly):
            row_split_dataonly = re.split(r'\t', row)[1:]
            expression_data[idx_row,:] = [float(val) for val in row_split_dataonly]
    return gene_labels, celltype_labels, expression_data


def binarize_data(xi):
    return 1.0 * np.where(xi > 0, 1, -1)  # mult by 1.0 to cast as float


def reduce_gene_set(xi, gene_labels):  # TODO: my removal ends with 1339 left but theirs with 1337 why?
    genes_to_remove = []
    for row_idx, row in enumerate(xi):
        if all(map(lambda x: x == row[0], row)):
            genes_to_remove.append(row_idx)
    reduced_gene_labels = [gene_labels[idx] for idx in xrange(len(xi)) if idx not in genes_to_remove]
    reduced_xi = np.array([row for idx, row in enumerate(xi) if idx not in genes_to_remove])
    return reduced_gene_labels, reduced_xi


def memory_corr_matrix_and_inv(xi):
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


def singlecell_simsetup(flag_prune_intxn_matrix=False):
    gene_labels, celltype_labels, xi = load_singlecell_data()
    if FLAG_BOOL:
        xi = binarize_data(xi)
    if FLAG_REMOVE_DUPES:
        assert FLAG_BOOL
        gene_labels, xi = reduce_gene_set(xi, gene_labels)
    a, a_inv = memory_corr_matrix_and_inv(xi)
    j = interaction_matrix(xi, a_inv, method=METHOD, flag_prune_intxn_matrix=flag_prune_intxn_matrix)
    eta = predictivity_matrix(xi, a_inv)
    return gene_labels, celltype_labels, len(gene_labels), len(celltype_labels), xi, a, a_inv, j, eta


# DEFINE SIMULATION CONSTANTS IN ISOLATED SETUP CALL
if FLAG_PRUNE_INTXN_MATRIX:
    print "Note FLAG_PRUNE_INTXN_MATRIX is True with ratio %.2f" % J_RANDOM_DELETE_RATIO
GENE_LABELS, CELLTYPE_LABELS, N, P, XI, A, A_INV, J, ETA = singlecell_simsetup(flag_prune_intxn_matrix=FLAG_PRUNE_INTXN_MATRIX)
CELLTYPE_ID = {k: v for v, k in enumerate(CELLTYPE_LABELS)}
GENE_ID = {k: v for v, k in enumerate(GENE_LABELS)}
