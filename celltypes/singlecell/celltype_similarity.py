import csv
import numpy as np
import os

from singlecell_constants import DATADIR, MEMORIESDIR


def load_external_celltype(csvpath):
    # TODO test
    gene_labels_raw = []
    gene_states_raw = []
    with open(csvpath, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            assert len(r) == 2
            gene_labels_raw.append(r[0])
            gene_states_raw.append(float(r[1]))
    return gene_labels_raw, gene_states_raw


def reduce_memories_to_matching_genes(simsetup, gene_labels_raw, gene_states_raw):
    # TODO
    print 'Loaded gene count from external celltype:',
    print 'Number of genes which match simsetup gene list:',
    external_gene_labels_sorted = None
    external_gene_states_sorted = None
    return xi_truncated


def score_similarity(external_celltype, memories_npz):
    gene_labels_raw, gene_states_raw = load_external_celltype(external_celltype)
    return


if __name__ == '__main__':
    # TODO get dermal fibroblast from mehta 2014
    # TODO implement and test functions
    # TODO separate grid loading
    external_celltypes_dir = DATADIR + os.sep + 'misc' + os.sep + 'external_celltypes'
    external_celltype = external_celltypes_dir + os.sep + '2014mehta_dermal_fibroblast.csv'
    memories_npz = MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_A_TFonly.npz'
    print 'Scoring similarity between %s and celltypes in %s' % (external_celltype, memories_npz)
    score_similarity(external_celltype, memories_npz)
