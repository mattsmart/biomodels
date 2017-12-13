import numpy as np
import re

from singlecell_constants import METHOD, FLAG_BOOL, ZSCORE_DATAFILE

"""
Conventions follow from Lang & Mehta 2014, PLOS Comp. Bio
- note the memory matrix is transposed throughout here (dim N x p instead of dim p x N)
"""


def load_singlecell_data(zscore_datafile=ZSCORE_DATAFILE):
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
    return np.where(xi > 0, 1, -1)


def memory_corr_matrix_and_inv(xi):
    corr_matrix = np.dot(xi.T, xi) / len(xi)
    return corr_matrix, np.linalg.inv(corr_matrix)


def interaction_matrix(xi, corr_inv, method):
    if method == "hopfield":
        return np.dot(xi, xi.T) / len(xi)                         # note not sure if factor 1/N needed
    elif method == "projection":
        return reduce(np.dot, [xi, corr_inv, xi.T]) / len(xi)     # note not sure if factor 1/N needed
    else:
        raise ValueError("method arg invalid, must be one of %s" % ["projection", "hopfield"])


def predictivity_matrix(xi, corr_inv):
    return np.dot(corr_inv, xi.T) / len(xi)  # eta_ij is the "predictivity" of TF i in cell fate j


def singlecell_simsetup():
    gene_labels, celltype_labels, xi = load_singlecell_data()
    if FLAG_BOOL:
        xi = binarize_data(xi)
    a, a_inv = memory_corr_matrix_and_inv(xi)
    j = interaction_matrix(xi, a_inv, method=METHOD)
    eta = predictivity_matrix(xi, a_inv)
    return gene_labels, celltype_labels, len(gene_labels), len(celltype_labels), xi, a, a_inv, j, eta


# DEFINE SIMULATION CONSTANTS IN ISOLATED SETUP CALL
GENE_LABELS, CELLTYPE_LABELS, N, P, XI, A, A_INV, J, ETA = singlecell_simsetup()
CELLTYPE_ID = {k: v for v, k in enumerate(CELLTYPE_LABELS)}
GENE_ID = {k: v for v, k in enumerate(GENE_LABELS)}
