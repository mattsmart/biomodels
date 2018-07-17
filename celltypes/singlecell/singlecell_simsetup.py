import numpy as np

from singlecell_constants import METHOD, FLAG_BOOL, DEFAULT_MEMORIES_NPZPATH, J_RANDOM_DELETE_RATIO, FLAG_PRUNE_INTXN_MATRIX
from singlecell_linalg import memory_corr_matrix_and_inv, interaction_matrix, predictivity_matrix
from dataprocess.data_standardize import load_npz_of_arr_genes_cells

"""
Conventions follow from Lang & Mehta 2014, PLOS Comp. Bio
- note the memory matrix is transposed throughout here (dim N x p instead of dim p x N)
"""


def singlecell_simsetup(flag_prune_intxn_matrix=False, npzpath=DEFAULT_MEMORIES_NPZPATH):
    """
    gene_labels, celltype_labels, xi = load_singlecell_data()
    """
    xi, gene_labels, celltype_labels = load_npz_of_arr_genes_cells(npzpath, verbose=True)
    gene_labels = gene_labels.tolist()
    celltype_labels = celltype_labels.tolist()
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
