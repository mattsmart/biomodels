import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell_constants import DATADIR, MEMORIESDIR
from singlecell_simsetup import singlecell_simsetup
from singlecell_simsetup_query import collect_mygene_hits, write_genelist_id_csv, read_gene_list_csv, check_target_in_gene_id_dict
from singlecell_visualize import plot_as_radar, plot_as_bar


def write_celltype_csv(gene_labels, celltype_state, csvpath):
    with open(csvpath, 'w') as f:
        for idx, label in enumerate(gene_labels):
            line = '%s,%d\n' % (label, celltype_state[idx])
            f.write(line)
    return csvpath


def load_external_celltype(csvpath):
    gene_labels_raw = []
    gene_states_raw = []
    with open(csvpath, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            assert len(r) == 2
            gene_labels_raw.append(r[0])
            gene_states_raw.append(float(r[1]))
    return gene_labels_raw, gene_states_raw


def build_entrez_synonyms_for_celltype(gene_labels_raw, entrez_name='entrez_id_ext_celltype.csv'):
    entrez_path = DATADIR + os.sep + 'misc' + os.sep + 'genelist_entrezids' + os.sep + entrez_name
    if os.path.exists(entrez_path):
        print "Warning, path %s exists" % entrez_path
        print "Delete file to have it be remade (using existing one)"
    else:
        gene_hits, hitcounts = collect_mygene_hits(gene_labels_raw)
        write_genelist_id_csv(gene_labels_raw, gene_hits, outpath=entrez_path)
    return entrez_path


def truncate_celltype_data(matches, simsetup, gene_labels_raw, gene_states_raw):
    # TODO
    print 'Loaded gene count from external celltype:', len(gene_labels_raw)
    print 'Loaded gene count from memories:', len(simsetup['GENE_LABELS'])
    print 'Number of matches, with possible redundancies:', len(matches)
    print 'NOTE: choosing FIRST gene match in cases of degenerate matches'
    new_memory_rows = []
    new_celltype_rows = []

    count=0

    for match in matches:
        matched_memory_gene = match[0]
        matched_celltype_gene = match[1]
        matched_memory_gene_idx = simsetup['GENE_ID'][matched_memory_gene]
        matched_celltype_gene_idx = gene_labels_raw.index(matched_celltype_gene)
        if matched_celltype_gene_idx in new_celltype_rows or matched_memory_gene_idx in new_memory_rows:
            continue
        else:
            new_memory_rows.append(matched_memory_gene_idx)
            new_celltype_rows.append(matched_celltype_gene_idx)
    assert len(new_memory_rows) == len(new_celltype_rows)
    print 'Number of genes which match simsetup gene list:', len(new_memory_rows)
    xi_truncated = simsetup['XI'][new_memory_rows, :]
    ext_gene_states_truncated = np.array(gene_states_raw)[new_celltype_rows]
    print xi_truncated.shape, ext_gene_states_truncated.shape
    return xi_truncated, ext_gene_states_truncated


def score_similarity(external_celltype, memories_npz, memories_entrez_path, celltype_name='unspecified_celltype'):
    # load external celltype gene labels and states
    gene_labels_raw, gene_states_raw = load_external_celltype(external_celltype)
    # build synonym list from the gene labels
    ext_entrez_path = build_entrez_synonyms_for_celltype(gene_labels_raw,
                                                          entrez_name='entrez_id_ext_celltype_%s.csv' % celltype_name)
    # reload synonym lists for memories and ext celltype
    target_genes_id = read_gene_list_csv(ext_entrez_path, aliases=True)
    memories_genes_id = read_gene_list_csv(memories_entrez_path, aliases=True)
    # compare synonym lists to get matches
    matches = check_target_in_gene_id_dict(memories_genes_id, target_genes_id,
                                           outpath='tmp_matches_%s_vs_memories.txt' % (celltype_name))
    # use matches to truncate ext gene states to match truncated memory states and
    simsetup = singlecell_simsetup(npzpath=memories_npz)
    xi_truncated, ext_gene_states_truncated = truncate_celltype_data(matches, simsetup, gene_labels_raw, gene_states_raw)
    # use truncated data to score overlaps
    overlaps = np.dot(xi_truncated.T, ext_gene_states_truncated)  # TODO div by num truncated genes? make sure shapes match too
    for idx, label in enumerate(simsetup['CELLTYPE_LABELS']):
        print "Celltype %s, overlap %.3f" % (label, overlaps[idx])
    # plot overlaps
    plot_as_bar(overlaps, simsetup['CELLTYPE_LABELS'])
    plt.title('Overlap between %s and loaded memories (num matching genes: %d)' % (celltype_name, len(ext_gene_states_truncated)))
    plt.savefig(os.path.join(os.path.dirname(external_celltype), 'score_%s_vs_mems.png' % (celltype_name)))
    plt.show()
    return


if __name__ == '__main__':
    create_ext_celltype = False
    score_ext_celltype = True
    score_reference = True

    # local constants
    external_celltypes_dir = DATADIR + os.sep + 'misc' + os.sep + 'external_celltypes'

    if create_ext_celltype:
        celltype_choice = 'fibroblast - skin'
        csvpath = external_celltypes_dir + os.sep + '2014mehta_dermal_fibroblast.csv'
        memories_npz = MEMORIESDIR + os.sep + '2014_mehta_mems_genes_types_boolean_compressed_pruned_A.npz'
        simsetup = singlecell_simsetup(npzpath=memories_npz)
        gene_labels = simsetup['GENE_LABELS']
        celltype_choice_idx = simsetup['CELLTYPE_ID'][celltype_choice]
        celltype_state = simsetup['XI'][:, celltype_choice_idx]
        write_celltype_csv(gene_labels, celltype_state, csvpath)

    if score_ext_celltype:
        # settings
        ext_celltype_name = 'dermal_fibroblast'
        external_celltype = external_celltypes_dir + os.sep + '2014mehta_dermal_fibroblast.csv'
        memories_npz = MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_A_TFonly.npz'
        memories_entrez_path = DATADIR + os.sep + 'misc' + os.sep + 'genelist_entrezids' + os.sep + 'entrez_id_2018scMCA_pruned_TFonly.csv'
        # scoring
        print 'Scoring similarity between %s and celltypes in %s...' % (external_celltype, memories_npz)
        score_similarity(external_celltype, memories_npz, memories_entrez_path, celltype_name=ext_celltype_name)

    if score_reference:
        print "TODO"