import numpy as np
import os

from data_standardize import parse_exptdata, load_npz_of_arr_genes_cells, save_npz_of_arr_genes_cells, \
                               load_npz_of_arr_genes_cells, load_cluster_labels, prune_boring_rows
from singlecell.singlecell_constants import DATADIR

# TODO pass metadata to all functions?
# TODO test and optimize build_basin_states
# TODO build remaining functions + unit tests


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


def prune_memories_genes(npzpath, save=True):
    rows_to_delete, mem_arr, genes, clusters = prune_boring_rows(npzpath, save=save)
    return rows_to_delete, mem_arr, genes, clusters


def prune_cluster_dict(cluster_dict, rows_to_delete):
    pruned_cluster_dict = {k: 0 for k in cluster_dict.keys()}
    for k in xrange(len(cluster_dict.keys())):
        cluster_data = cluster_dict[k]
        pruned_cluster_dict[k] = np.delete(cluster_data, rows_to_delete, axis=0)
    return pruned_cluster_dict


if __name__ == '__main__':
    # run flags
    datadir = DATADIR + os.sep + "2018_scMCA"
    flag_load_raw = False
    flag_prune_mems = False
    flag_prune_rawdata = True

    # options
    verbose = True
    binarize_method = "by_gene"  # either 'by_cluster', 'by_gene'
    memory_method = "default"
    basinscore_method = "trajectories"  # either 'trajectories', 'crawler'

    rawdata_npzpath = datadir + os.sep + 'arr_genes_cells_withcluster_compressed.npz'

    if flag_load_raw:
        arr, genes, cells = load_npz_of_arr_genes_cells(rawdata_npzpath)
        cluster_dict, metadata = parse_exptdata(arr, genes, verbose=verbose)
        binarized_cluster_dict = binarize_cluster_dict(cluster_dict, metadata, binarize_method=binarize_method)
        memory_array = binary_cluster_dict_to_memories(binarized_cluster_dict, metadata, memory_method=memory_method)

    if flag_prune_mems:
        rawmems_npzpath = DATADIR + os.sep + 'mems_genes_types_compressed.npz'
        rows_to_delete, memory_array, genes, clusters = prune_memories_genes(rawmems_npzpath, save=True)  # TODO prune cluster dict based on this pruning...
        np.savetxt(DATADIR + os.sep + 'rows_to_delete_A.txt', np.array(rows_to_delete), delimiter=",", fmt="%d")  # note these are indexed with 0 a gene not 'cluster_id'
        binarized_cluster_dict = prune_cluster_dict(binarized_cluster_dict, rows_to_delete)

        prunedmems_npzpath = DATADIR + os.sep + 'mems_genes_types_compressed_pruned.npz'
        rows_to_delete, memory_array, genes, clusters = prune_memories_genes(prunedmems_npzpath, save=True)  # TODO prune cluster dict based on this pruning...
        np.savetxt(DATADIR + os.sep + 'rows_to_delete_AB.txt', np.array(rows_to_delete), delimiter=",", fmt="%d")  # note these are indexed with 0 a gene not 'cluster_id'
        binarized_cluster_dict = prune_cluster_dict(binarized_cluster_dict, rows_to_delete)

    if flag_prune_rawdata:
        #TODO fix pruning so its all done at once (boring all off/on and dupes, then save single pruned rows file
        rows_to_delete = np.loadtxt(DATADIR + os.sep + 'rows_to_delete_AB.txt')  # note these are indexed with 0 a gene not 'cluster_id'
        rows_to_delete_increment_for_clusterrow = [i+1 for i in rows_to_delete]
        _, arr, genes, cells = prune_boring_rows(rawdata_npzpath, specified_rows=rows_to_delete_increment_for_clusterrow)
