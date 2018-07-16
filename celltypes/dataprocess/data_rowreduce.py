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


def prune_boring_rows(npzpath, specified_rows=None, save=True):
    """
    Delete rows from array and corresponding genes that are self-duplicates
    NOTE: very similar to reduce_gene_set(xi, gene_labels)
    """
    arr, genes, cells = load_npz_of_arr_genes_cells(npzpath)
    print "CHECK FIRST ROW NOT CLUSTER ROW:", genes[0]
    if specified_rows is None:
        # collect rows to delete
        rows_duplicates = np.all(arr.T == arr.T[0,:], axis=0)
        rows_to_delete = np.array([idx for idx, val in enumerate(rows_duplicates) if val])
        # note pruned rows
        print "number of self-duplicate rows:", len(rows_to_delete)
        print rows_to_delete
        print rows_to_delete[0:10]
    else:
        rows_to_delete = np.array(specified_rows)
    # adjust genes and arr contents
    print "Orig shape arr, genes, cells:", arr.shape, genes.shape, cells.shape
    arr = np.delete(arr, rows_to_delete, axis=0)
    genes = np.delete(genes, rows_to_delete)  # TODO should have global constant for this mock gene label
    print "New shape arr, genes, cells:", arr.shape, genes.shape, cells.shape
    # save and return data
    if save:
        print "saving pruned arrays..."
        datadir = os.path.abspath(os.path.join(npzpath, os.pardir))
        base = os.path.basename(npzpath)
        basestr = os.path.splitext(base)[0]
        savestr = basestr + '_pruned.npz'
        np.savez_compressed(datadir + os.sep + savestr, arr=arr, genes=genes, cells=cells)
    return rows_to_delete, arr, genes, cells


def prune_duplicate_rows(npzpath, specified_rows=None, save=True):
    """
    Find duplicate rows (i.e. row i and row j are the same)
    Retain only one copy of each duplicate row (the last one, via np.unique())
    """
    arr, genes, cells = load_npz_of_arr_genes_cells(npzpath)
    num_rows, num_cols = arr.shape
    print "CHECK FIRST ROW NOT CLUSTER ROW:", genes[0]

    if specified_rows is None:
        # collect rows to delete
        _, unique_indices = np.unique(arr, return_index=True, axis=0)
        rows_to_delete = np.array(list(set(range(num_rows)) - set(unique_indices)))
        # note pruned rows
        print "number of duplicated rows (num to delete):", len(rows_to_delete)
        print rows_to_delete
        print rows_to_delete[0:10]
    else:
        rows_to_delete = np.array(specified_rows)
    # adjust genes and arr contents
    print "Orig shape arr, genes, cells:", arr.shape, genes.shape, cells.shape
    arr = np.delete(arr, rows_to_delete, axis=0)
    genes = np.delete(genes, rows_to_delete)  # TODO should have global constant for this mock gene label
    print "New shape arr, genes, cells:", arr.shape, genes.shape, cells.shape
    # save and return data
    if save:
        print "saving (no dupes) pruned arrays..."
        datadir = os.path.abspath(os.path.join(npzpath, os.pardir))
        base = os.path.basename(npzpath)
        basestr = os.path.splitext(base)[0]
        savestr = basestr + '_nodupes.npz'
        np.savez_compressed(datadir + os.sep + savestr, arr=arr, genes=genes, cells=cells)
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
    flag_prune_boring_rows = False
    flag_prune_duplicate_rows = True

    if flag_prune_boring_rows:
        compressed_file = datadir + os.sep + "arr_genes_cells_raw_compressed.npz"
        prune_boring_rows(compressed_file)

    if flag_prune_duplicate_rows:
        compressed_file = datadir + os.sep + "mems_genes_types_compressed_pruned.npz"
        prune_duplicate_rows(compressed_file)
