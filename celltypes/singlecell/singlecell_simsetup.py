import os

from singlecell_constants import NETWORK_METHOD, DEFAULT_MEMORIES_NPZPATH, J_RANDOM_DELETE_RATIO, \
    FLAG_PRUNE_INTXN_MATRIX, MEMORIESDIR
from singlecell_linalg import memory_corr_matrix_and_inv, interaction_matrix, predictivity_matrix
from dataprocess.data_standardize import load_npz_of_arr_genes_cells, save_npz_of_arr_genes_cells
from dataprocess.unfolding_csv_to_npz import load_npz_of_arr_genes_cells_signals

"""
Conventions follow from Lang & Mehta 2014, PLOS Comp. Bio
- note the memory matrix is transposed throughout here (dim N x p instead of dim p x N)
"""


def singlecell_simsetup(flag_prune_intxn_matrix=FLAG_PRUNE_INTXN_MATRIX, npzpath=DEFAULT_MEMORIES_NPZPATH, unfolding=False):
    """
    gene_labels, celltype_labels, xi = load_singlecell_data()
    """
    assert NETWORK_METHOD in ["projection", "hopfield"]
    # comments
    if flag_prune_intxn_matrix:
        print "Note FLAG_PRUNE_INTXN_MATRIX is True with ratio %.2f" % J_RANDOM_DELETE_RATIO
    # unfolding block
    if unfolding:
        print "Using unfolding npz"
        xi, gene_labels, celltype_labels, signals = load_npz_of_arr_genes_cells_signals(npzpath, verbose=True)
        FIELD_SEND = signals
    else:
        xi, gene_labels, celltype_labels = load_npz_of_arr_genes_cells(npzpath, verbose=True)
        FIELD_SEND = None
    # data processing into sim object
    gene_labels = gene_labels.tolist()
    celltype_labels = celltype_labels.tolist()
    a, a_inv = memory_corr_matrix_and_inv(xi)
    j = interaction_matrix(xi, a_inv, method=NETWORK_METHOD, flag_prune_intxn_matrix=flag_prune_intxn_matrix)
    eta = predictivity_matrix(xi, a_inv)
    if NETWORK_METHOD == "hopfield":
        #a = np.eye(len(celltype_labels))      # identity p x p
        #a_inv = np.eye(len(celltype_labels))  # identity p x p
        #eta = np.copy(xi)
        print "Warning, NOT changing correlation matrix to identity"
    celltype_id = {k: v for v, k in enumerate(celltype_labels)}
    gene_id = {k: v for v, k in enumerate(gene_labels)}
    # store in sim object (currently just a dict)
    simsetup = {
        'memories_path': npzpath,
        'N': len(gene_labels),
        'P': len(celltype_labels),
        'GENE_LABELS': gene_labels,
        'CELLTYPE_LABELS': celltype_labels,
        'GENE_ID': gene_id,
        'CELLTYPE_ID': celltype_id,
        'XI': xi,
        'A': a,
        'A_INV': a_inv,
        'J': j,
        'ETA': eta,
        'NETWORK_METHOD': NETWORK_METHOD,
        'FIELD_SEND': FIELD_SEND,
        'bool_gpu': False
    }
    return simsetup


def unpack_simsetup(simsetup):
    N = simsetup['N']
    P = simsetup['P']
    GENE_LABELS = simsetup['GENE_LABELS']
    CELLTYPE_LABELS = simsetup['CELLTYPE_LABELS']
    GENE_ID = simsetup['GENE_ID']
    CELLTYPE_ID = simsetup['CELLTYPE_ID']
    XI = simsetup['XI']
    A = simsetup['A']
    A_INV = simsetup['A_INV']
    J = simsetup['J']
    ETA = simsetup['ETA']
    return N, P, GENE_LABELS, CELLTYPE_LABELS, GENE_ID, CELLTYPE_ID, XI, A, A_INV, J, ETA


if __name__ == '__main__':

    print_genes = True
    print_celltypes = True
    print_gene_excpression_row = False
    npzpath_override = True
    npzpath_alternate = MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_A_TFonly.npz'
    # print block
    if npzpath_override:
        simsetup = singlecell_simsetup(npzpath=npzpath_alternate)
    else:
        simsetup = singlecell_simsetup()
    if print_gene_excpression_row:
        gene_name = 'S100a4'
        gene_int = simsetup['GENE_ID'][gene_name]
    if print_genes:
        print 'Genes:'
        for idx, label in enumerate(simsetup['GENE_LABELS']):
            print idx, label
    if print_celltypes:
        print 'Celltypes:'
        for idx, label in enumerate(simsetup['CELLTYPE_LABELS']):
            if print_gene_excpression_row:
                print idx, label, '|', gene_name, simsetup['XI'][gene_int, idx]
            else:
                print idx, label

    edit_npz = False
    # edit npz block
    if edit_npz:
        npz_to_edit = MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_AB.npz'
        npz_outname = MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_AB_edit.npz'
        xi, gene_labels, celltype_labels = load_npz_of_arr_genes_cells(npz_to_edit, verbose=True)
        # edits go here
        celltype_labels[58] = 'Mixed NK T cell'
        save_npz_of_arr_genes_cells(npz_outname, xi, gene_labels, celltype_labels)
        xi, gene_labels, celltype_labels = load_npz_of_arr_genes_cells(npz_outname, verbose=True)
        if print_celltypes:
            print 'Celltypes:'
            for idx, label in enumerate(celltype_labels):
                print idx, label
