import numpy as np
import os

from data_settings import DATADIR
from data_standardize import load_npz_of_arr_genes_cells

"""
Script to reduce row count (number of genes) in single-cell RNA expression data (N genes x M cells)
- prune boring rows: delete rows which are all on or all off
- prune duplicate rows: delete rows which are copies of other rows
TODO: less naive dimension reduction (PCA, others)
"""

def prune_rows(npzpath, specified_rows=None, save_pruned=True, save_rows=True):
    """
    Delete rows from array and corresponding genes that are self-duplicates
    NOTE: very similar to reduce_gene_set(xi, gene_labels)
    """
    arr, genes, cells = load_npz_of_arr_genes_cells(npzpath)
    num_rows, num_cols = arr.shape
    print "CHECK FIRST ROW NOT CLUSTER ROW:", genes[0]
    if specified_rows is None:
        # collect rows to delete (A - self-duplicate rows all on / all off)
        rows_duplicates = np.all(arr.T == arr.T[0,:], axis=0)
        rows_to_delete_self_dup = set([idx for idx, val in enumerate(rows_duplicates) if val])
        print "number of self-duplicate rows:", len(rows_to_delete_self_dup)

        # collect rows to delete (B - rows which are copies of other rows)
        _, unique_indices = np.unique(arr, return_index=True, axis=0)
        rows_to_delete_dupe = set(range(num_rows)) - set(unique_indices)
        print "number of duplicated rows (num to delete):", len(rows_to_delete_dupe)

        rows_to_delete = np.array(list(rows_to_delete_dupe.union(rows_to_delete_self_dup)))
    else:
        save_rows = False
        rows_to_delete = np.array(specified_rows)
    # adjust genes and arr contents
    print "Orig shape arr, genes, cells:", arr.shape, genes.shape, cells.shape
    arr = np.delete(arr, rows_to_delete, axis=0)
    genes = np.delete(genes, rows_to_delete)  # TODO should have global constant for this mock gene label
    print "New shape arr, genes, cells:", arr.shape, genes.shape, cells.shape
    # save and return data
    datadir = os.path.abspath(os.path.join(npzpath, os.pardir))
    if save_pruned:
        print "saving pruned arrays..."
        base = os.path.basename(npzpath)
        basestr = os.path.splitext(base)[0]
        savestr = basestr + '_pruned.npz'
        np.savez_compressed(datadir + os.sep + savestr, arr=arr, genes=genes, cells=cells)
    if save_rows:
        np.savetxt(datadir + os.sep + 'rows_to_delete.txt', rows_to_delete, delimiter=",", fmt="%d")
    return rows_to_delete, arr, genes, cells


def reduce_gene_set(xi, gene_labels):  # TODO: my removal ends with 1339 left but theirs with 1337 why?
    """
    NOTE: very similar to prune_boring_rows(...)
    """
    genes_to_remove = []
    for row_idx, row in enumerate(xi):
        if all(map(lambda x: x == row[0], row)):
            genes_to_remove.append(row_idx)
    reduced_gene_labels = [gene_labels[idx] for idx in xrange(len(xi)) if idx not in genes_to_remove]
    reduced_xi = np.array([row for idx, row in enumerate(xi) if idx not in genes_to_remove])
    return reduced_gene_labels, reduced_xi


if __name__ == '__main__':
    datadir = DATADIR
    flag_prune_rows = False
    flag_prune_duplicate_rows = False

    if flag_prune_rows:
        compressed_file = datadir + os.sep + "arr_genes_cells_raw_compressed.npz"
        prune_rows(compressed_file)
