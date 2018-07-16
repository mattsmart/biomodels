import numpy as np
import os

from data_cluster import load_cluster_labels, attach_cluster_id_arr_manual
from data_rowreduce import prune_rows
from data_settings import DATADIR, PIPELINES_VALID, WITHCLUSTERS_2018SCMCA, NPZ_2018SCMCA_MEMS, NPZ_2018SCMCA_ORIG, \
                          NPZ_2018SCMCA_ORIG_WITHCLUSTER, NPZ_2014MEHTA_ORIG
from data_standardize import parse_exptdata, save_npz_of_arr_genes_cells, load_npz_of_arr_genes_cells


"""
Purpose: process standardized expression data (i.e. converted to npz of arr, genes, cells)
 - cluster data, or load in clustered results and attach it to first row of gene, expression in the npz
 - save clustered raw data in standard npz format (npz of arr, genes, cells)
 - convert raw data into "cluster dict": dictionary that maps cluster data to submatrix of genes x cells
 - binarize data within each cluster dict
 - create binarized cluster dict
 - from binarized cluster dict: create "memory" / "cell type" matrix (get representative column from each cluster)
 - save memory matrix in standard npz format (npz of mems, genes, types)
 - reduce row number with various pruning techniques
 - save total row reduction in file "removed_rows.txt"
 - save reduced memory matrix in standard npz format (npz of mems, genes, types)
 - use "removed_rows.txt" to delete rows of original raw data 
 - save reduced clustered raw data in standard npz format (npz of arr, genes, cells)
 - save reduced unclustered raw data in standard npz format (npz of arr, genes, cells)
Main output:
 - reduced memory matrix is used as input to singlecell module
"""

# TODO pass metadata to all functions?
# TODO test and optimize build_basin_states
# TODO build remaining functions + unit tests
# TODO have report script which stores all processing flags/choices/order
# TODO maybe have rundir for results of each proc run
# TODO how to save cluster dict? as npz?


def binarize_data(xi):
    return 1.0 * np.where(xi > 0, 1, -1)  # mult by 1.0 to cast as float


def binarize_cluster_dict(cluster_dict, metadata, binarize_method="by_gene"):
    """
    Args:
        - cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - binarize_method: options for different binarization methods: by_cluster or by_gene (default)
    Returns:
        - binarized_cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
    """
    assert binarize_method in ['by_cluster', 'by_gene']
    num_clusters = metadata['num_clusters']

    print num_clusters, np.max(cluster_dict.keys()), cluster_dict[0].shape

    binarize_cluster_dict = {}
    if binarize_method == 'by_gene':
        for k in xrange(num_clusters):
            cluster_data = cluster_dict[k]
            min_gene_vals = np.amin(cluster_data, axis=1)  # min value each gene has over all cells in the cluster
            max_gene_vals = np.amax(cluster_data, axis=1)
            mids = 0.5 * (min_gene_vals - max_gene_vals)
            # TODO vectorize this
            binarized_cluster = np.zeros(cluster_data.shape)
            for idx in xrange(cluster_data.shape[0]):
                binarized_cluster[idx,:] = np.where(cluster_data[idx,:] > mids[idx], 1.0, -1.0)  # mult by 1.0 to cast as float
            binarize_cluster_dict[k] = binarized_cluster
    else:
        print "WARNING: binarize_method by_cluster is not stable (data too sparse)"
        for k in xrange(num_clusters):
            cluster_data = cluster_dict[k]
            min_val = np.min(cluster_data)
            max_val = np.max(cluster_data)
            mid = 0.5 * (max_val - min_val)
            binarized_cluster = 1.0 * np.where(cluster_data > mid, 1, -1)  # mult by 1.0 to cast as float
            binarize_cluster_dict[k] = binarized_cluster

    return binarize_cluster_dict


def binary_cluster_dict_to_memories(binarized_cluster_dict, metadata, memory_method="default", save=True):
    """
    Args:
        - binarized_cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - metadata: dict, mainly stores N x 1 array of 'gene_labels' for each row
        - memory_method: options for different memory processing algos
    Returns:
        - memory_array: i.e. xi matrix, will be N x K (one memory from each cluster)
    """
    num_genes = metadata['num_genes']
    #num_cells = metadata['num_cells']
    num_clusters = metadata['num_clusters']

    eps = 1e-4  # used to bias the np.sign(call) to be either 1 or -1 (breaks ties towards on state)
    memory_array = np.zeros((num_genes, num_clusters))
    for k in xrange(num_clusters):
        cluster_arr = binarized_cluster_dict[k]
        cluster_arr_rowsum = np.sum(cluster_arr, axis=1)
        memory_vec = np.sign(cluster_arr_rowsum + eps)
        memory_array[:,k] = memory_vec
    if save:
        npzpath = DATADIR + os.sep + 'mems_genes_types_compressed.npz'
        store_memories_genes_clusters(npzpath, memory_array, np.array(metadata['gene_labels']))
    return memory_array


def store_memories_genes_clusters(npzpath, mem_arr, genes):
    cluster_id = load_cluster_labels(DATADIR + os.sep + '2018_scMCA' + os.sep + 'SI_cluster_labels.csv')
    clusters = np.array([cluster_id[idx] for idx in xrange(len(cluster_id.keys()))])
    save_npz_of_arr_genes_cells(npzpath, mem_arr, genes, clusters)
    return


def load_memories_genes_clusters(npzpath):
    mem_arr, genes, clusters = load_npz_of_arr_genes_cells(npzpath, verbose=False)
    return mem_arr, genes, clusters


def prune_memories_genes(npzpath):
    rows_to_delete, mem_arr, genes, clusters = prune_rows(npzpath, save_pruned=True, save_rows=True)
    return rows_to_delete, mem_arr, genes, clusters


def prune_cluster_dict(cluster_dict, rows_to_delete):
    pruned_cluster_dict = {k: 0 for k in cluster_dict.keys()}
    for k in xrange(len(cluster_dict.keys())):
        cluster_data = cluster_dict[k]
        pruned_cluster_dict[k] = np.delete(cluster_data, rows_to_delete, axis=0)
    return pruned_cluster_dict


if __name__ == '__main__':

    # choose pipeline from PIPELINES_VALID
    pipeline = "2018_scMCA"
    assert pipeline in PIPELINES_VALID
    datadir = DATADIR + os.sep + pipeline

    if pipeline == "2014_mehta":
        flag_load_raw = True
        flag_prune_mems = True
        flag_prune_rawdata = True
        # options
        verbose = True

        # part 1: binarize standardized npz zscore data
        npzpath = NPZ_2014MEHTA_ORIG
        expression_data, genes, celltypes = load_npz_of_arr_genes_cells(npzpath, verbose=True)
        xi = binarize_data(expression_data)
        compressed_boolean = datadir + os.sep + "mems_genes_types_boolean_compressed.npz"
        save_npz_of_arr_genes_cells(compressed_boolean, xi, genes, celltypes)
        # part 3: load npz, prune, save
        rows_to_delete, xi, genes, celltypes = prune_rows(compressed_boolean, save_pruned=True, save_rows=True)

    elif pipeline == "2018_scMCA":
        flag_add_cluster_to_raw = True
        flag_build_cluster_dict_metadata = True
        flag_binarize_clusters = True
        flag_gen_memory_matrix = True
        flag_prune_mems = True
        flag_prune_raw = True
        # options
        verbose = True
        binarize_method = "by_gene"  # either 'by_cluster', 'by_gene'
        memory_method = "default"

        # (1) ensure raw data npz created (using data_standardize.npz)
        # (2) cluster raw data npz (add known clusters)  -> raw npz wth cluster row
        if flag_add_cluster_to_raw:
            clusterpath = WITHCLUSTERS_2018SCMCA
            arr, genes, cells = attach_cluster_id_arr_manual(NPZ_2018SCMCA_ORIG, clusterpath, save=True, one_indexed=True)
        # (3) process npz of raw data with clusters  -> cluster_dict, metadata
        if flag_build_cluster_dict_metadata:
            if not flag_add_cluster_to_raw:
                arr, genes, cells = load_npz_of_arr_genes_cells(WITHCLUSTERS_2018SCMCA)
            cluster_dict, metadata = parse_exptdata(arr, genes, verbose=verbose)
        # (4) binarize cluster dct -> binarized_cluster_dict
        if flag_binarize_clusters:
            if not flag_build_cluster_dict_metadata:
                arr, genes, cells = load_npz_of_arr_genes_cells(WITHCLUSTERS_2018SCMCA)
                cluster_dict, metadata = parse_exptdata(arr, genes, verbose=verbose)
            binarized_cluster_dict = binarize_cluster_dict(cluster_dict, metadata, binarize_method=binarize_method)
        # (5) -> boolean memory matrix, mem genes types npz
        if flag_gen_memory_matrix:
            if not flag_binarize_clusters:
                arr, genes, cells = load_npz_of_arr_genes_cells(WITHCLUSTERS_2018SCMCA)
                cluster_dict, metadata = parse_exptdata(arr, genes, verbose=verbose)
                binarized_cluster_dict = binarize_cluster_dict(cluster_dict, metadata, binarize_method=binarize_method)
            _ = binary_cluster_dict_to_memories(binarized_cluster_dict, metadata, memory_method=memory_method)
        # (6) prune mems -> boolean memory matrix (pruned), mem genes types npz
        if flag_prune_mems:
            rawmems_npzpath = NPZ_2018SCMCA_MEMS
            rows_to_delete, memory_array, genes, clusters = prune_memories_genes(rawmems_npzpath)  # TODO prune cluster dict based on this pruning...
        # (7) prune raw data -> boolean memory matrix (pruned), mem genes types npz
        if flag_prune_raw:
            if not flag_prune_mems:
                rows_to_delete = np.loadtxt(datadir + os.sep + "rows_to_delete.txt", delimiter=',')
            binarized_cluster_dict = prune_cluster_dict(binarized_cluster_dict, rows_to_delete)
            rows_to_delete = np.loadtxt(datadir + os.sep + 'rows_to_delete.txt')
            rows_to_delete_increment_for_clusterrow = [i+1 for i in rows_to_delete]
            _, _, _, _ = prune_rows(NPZ_2018SCMCA_ORIG, specified_rows=rows_to_delete, save_pruned=True, save_rows=False)
            _, _, _, _ = prune_rows(NPZ_2018SCMCA_ORIG_WITHCLUSTER, specified_rows=rows_to_delete_increment_for_clusterrow,
                                    save_pruned=True, save_rows=False)

    else:
        # TODO
        flag_load_raw = True
        flag_prune_mems = True
        flag_prune_rawdata = True
        # options
        verbose = True
        binarize_method = "by_gene"  # either 'by_cluster', 'by_gene'
        memory_method = "default"
        print "No misc processing implemented"
