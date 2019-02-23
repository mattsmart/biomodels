import numpy as np
import os

from singlecell_constants import MEMORIESDIR
from singlecell_linalg import memory_corr_matrix_and_inv, interaction_matrix, predictivity_matrix
from singlecell_simsetup import singlecell_simsetup, unpack_simsetup


if __name__ == '__main__':
    print_genes = True
    print_celltypes = True
    print_gene_expression_row = False
    npzpath_override = False
    npzpath_alternate = MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_A_TFonly.npz'
    unfolding = True
    # print block
    if npzpath_override:
        simsetup = singlecell_simsetup(npzpath=npzpath_alternate, unfolding=unfolding)
    else:
        simsetup = singlecell_simsetup(unfolding=unfolding)
    if print_gene_expression_row:
        gene_name = 'S100a4'
        gene_int = simsetup['GENE_ID'][gene_name]
    if print_genes:
        print 'Genes:'
        for idx, label in enumerate(simsetup['GENE_LABELS']):
            print idx, label
    if print_celltypes:
        print 'Celltypes:'
        for idx, label in enumerate(simsetup['CELLTYPE_LABELS']):
            if print_gene_expression_row:
                print idx, label, '|', gene_name, simsetup['XI'][gene_int, idx]
            else:
                print idx, label

    print "\nLinalg Block\n"
    np.set_printoptions(precision=3, linewidth=110)
    N, P, GENE_LABELS, CELLTYPE_LABELS, GENE_ID, CELLTYPE_ID, XI, A, A_INV, J, ETA = unpack_simsetup(simsetup)
    print "mems:"
    for idx in xrange(P):
        print idx, XI[:, idx]

    q, r = np.linalg.qr(XI)
    print "\northogonal mem basis (up to a sign)"
    for idx in xrange(P):
        print idx, q[:, idx]

    print "\nNaive stabilizer"
    hstb_naive = np.sum(XI, axis=1)
    print hstb_naive
    print "memory alignment"
    print np.dot(hstb_naive, XI)
    print np.sum(np.dot(hstb_naive, XI))
    print "effect of stabilizing field on well depth"
    print [-np.dot(XI[:, i], hstb_naive) for i in xrange(P)]

    print "\nQR stabilizer"
    hstb_proj = np.sum(q, axis=1)
    print hstb_proj
    print "memory alignment"
    print np.dot(hstb_proj, XI)
    print np.sum(np.dot(hstb_proj, XI))
    print "effect of stabilizing field on well depth"
    print [-np.dot(XI[:, i], hstb_proj) for i in xrange(P)]

    print "\n\nANCHOR block"
    print "Going to extend memories with redundant ON genes ('housekeeping')"
    num_anchors = 4
    XI_EXTEND = np.vstack((XI, np.ones((num_anchors, P))))
    GENE_LABELS_EXTEND = GENE_LABELS + ['anchor_%d' % i for i in xrange(num_anchors)]
    # print
    print XI_EXTEND
    print GENE_LABELS_EXTEND
    # repeat linalg steps
    A_EXTEND, A_INV_EXTEND = memory_corr_matrix_and_inv(XI_EXTEND)
    J_EXTEND = interaction_matrix(XI_EXTEND, A_INV_EXTEND, method="projection", flag_prune_intxn_matrix=False)
    ETA_EXTEND = predictivity_matrix(XI_EXTEND, A_INV_EXTEND)
    # compare vs non-anchored versions
    print J
    print J_EXTEND[:N,:N]
    print J_EXTEND
    print J - J_EXTEND[:N,:N]