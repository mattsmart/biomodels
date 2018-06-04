import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell.singlecell_constants import DATADIR
from singlecell.singlecell_functions import hamiltonian, hamming, memory_corr_matrix_and_inv, interaction_matrix

# TODO pass metadata to all functions?
# TODO test and optimize read_exptdata_from_files
# TODO build remaining functions
# TODO change code file structure with expt data IO folder in constants or expt_constants?


def read_exptdata_from_files(dataname, labelname, datadir=DATADIR, verbose=True):
    """
    Args:
        - datapath: stores array of state data and cluster labels for each cell state (column)
        - labelpath: stores row names i.e. gene or PCA labels
        - datadir: dataname and labelname must be in same data directory
    Notes: data format may change with time
        - convention is first row stores cluster index, from 0 to np.max(row 0) == K - 1
        - future convention may be to store unique integer ID for each column corresponding to earlier in  pipeline
        - maybe also extract and pass metadata_dict info (i.e. K, N, M, filename information on pipeline)
    Returns:
        - cluster_dict: {cluster_idx: N x M array of raw cell states in the cluster (i.e. not binarized)}
        - metadata: dict, mainly stores N x 1 array of 'gene_labels' for each row
    """
    datapath = datadir + os.sep + dataname
    labelpath = datadir + os.sep + labelname
    # load data
    states_raw = np.loadtxt(datapath, delimiter=",", dtype=float)
    states_row0 = states_raw[0, :]
    states_truncated = states_raw[1:, :]
    num_genes, num_cells = states_truncated.shape  # aka N, M
    num_clusters = np.max(states_raw[0, :]) + 1
    if verbose:
        print "loading data from %s..." % datapath
        print "raw data dimension: %d x %d" % (states_raw.shape)
        print "cleaned data dimension: %d x %d" % (num_genes, num_genes)
        print "num_clusters is %d" % num_clusters

    # load labels
    gene_labels = np.loadtxt(labelpath, delimiter=",")
    assert gene_labels.shape[0] == num_genes
    assert gene_labels.shape[1] == 1
    if verbose:
        print "loading labels from %s..." % labelpath

    # prep cluster_dict
    cluster_dict = {}
    cluster_indices = {k: [] for k in xrange(num_clusters)}
    # TODO optimize this chunk if needed
    for cell in xrange(num_cells):
        for k in xrange(num_clusters):
            cluster_indices[k].append(cell)
            break
    if verbose:
        print "cluster_indices...\n", cluster_indices

    # build cluster dict
    for k in xrange(num_clusters):
        cluster_dict[k] = states_truncated.take(cluster_indices[k], axis=1)

    # fill metatadata dict
    metadata = {}
    metadata['gene_labels'] = gene_labels
    metadata['num_clusters'] = num_clusters
    metadata['K'] = num_clusters
    metadata['num_genes'] = num_genes
    metadata['N'] = num_genes
    metadata['num_cells'] = num_cells
    metadata['M'] = num_cells
    metadata['datapath'] = datapath
    metadata['labelpath'] = labelpath
    metadata['dataname'] = dataname
    metadata['labelname'] = labelname
    return cluster_dict, metadata


def binarize_cluster_dict(cluster_dict, metadata, binarize_method="default"):
    """
    Args:
        - cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - binarize_method: options for different binarization methods
    Returns:
        - binarized_cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
    """
    num_clusters = metadata['num_clusters']
    binarize_cluster_dict = {}
    for x in xrange(num_clusters):
        cluster_data = cluster_dict[k]
        min_val = np.min(cluster_data)
        max_val = np.max(cluster_data)
        mid = 0.5 * (max_val - min_val)  # TODO implement column wise process of this too, compare
        binarized_cluster = 1.0 * np.where(cluster_data > mid, 1, -1)  # mult by 1.0 to cast as float
        binarize_cluster_dict[k] = binarized_cluster
    return binarize_cluster_dict


def binary_cluster_dict_to_memories(binarized_cluster_dict, metadata, memory_method="default"):
    """
    Args:
        - binarized_cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - metadata: dict, mainly stores N x 1 array of 'gene_labels' for each row
        - memory_method: options for different memory processing algos
    Returns:
        - memory_array: i.e. xi matrix, will be N x K (one memory from each cluster)
    """
    num_genes = metadata['num_genes']
    num_cells = metadata['num_cells']
    num_clusters = metadata['num_clusters']
    memory_array = 0

    return memory_array


def is_energy_increase(intxn_matrix, memory_vec, data_vec):
    H_data = hamiltonian(data_vec, intxn_matrix=intxn_matrix)
    H_mem = hamiltonian(memory_vec, intxn_matrix=intxn_matrix)
    if H_data > H_mem:
        return True
    else:
        return False


def build_basin_states(intxn_matrix, memory_vec,
                       recurse_dist_d=0, recurse_basin_set=None, recurse_state_copy=None,
                       sites_flipped_already=None):
    """
    Args:
        - intxn_matrix: J_ij built from memory_matrix
        - memory_vec: column of the memory matrix
        - various recursive arguments
    Returns:
        - basin_set: dict of {num_flips: SET (not list) of states as Nx1 lists} comprising the basin
    """
    num_genes = intxn_matrix.shape[0]
    memory_vec_copy = [val for val in memory_vec]

    if recurse_basin_set is None:
        recurse_basin_set = {d: [] for d in xrange(num_genes)}
        recurse_basin_set = {0: [memory_vec_copy]}
        recurse_state_copy = memory_vec_copy
        sites_flipped_already = []

    hamming_dist = 0                                       # hamming distance
    size_basin_at_dist_d = len(recurse_basin_set[hamming_dist])    # number of states with hamming dist = d in the basin

    for site_idx in [val for val in xrange(num_genes) if val not in sites_flipped_already]:
        recurse_state_copy = [val for val in recurse_state_copy]
        recurse_state_copy[site_idx] = -1 * recurse_state_copy[site_idx]
        if is_energy_increase(intxn_matrix, memory_vec, recurse_state_copy):
            recurse_basin_set[recurse_dist_d].add(recurse_state_copy)
            recurse_dist_d += 1
            recurse_sites_flipped_already = [val for val in sites_flipped_already]
            recurse_sites_flipped_already.append(site_idx)
            build_basin_states(intxn_matrix, memory_vec,
                               recurse_dist_d=recurse_dist_d,
                               recurse_basin_set=recurse_basin_set,
                               recurse_state_copy=recurse_state_copy,
                               sites_flipped_already=recurse_sites_flipped_already)
        else:
            print 'ending recursion branch'
            return recurse_basin_set

    return recurse_basin_set


def get_basins_scores(memory_array, binarized_cluster_dict, metadata, basinscore_method="default"):
    """
    Args:
        - memory_array: i.e. xi matrix, will be N x K (one memory from each cluster)
        - binarized_cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - metadata: dict, mainly stores N x 1 array of 'gene_labels' for each row
        - basinscore_method: options for different basin scoring algos
    Returns:
        - score_dict: {k: M x 1 array for k in 0 ... K-1 (i.e. cluster index)}
    """
    num_genes = metadata['num_genes']
    num_cells = metadata['num_cells']
    num_clusters = metadata['num_clusters']

    def basin_score_pairwise(basin_k, memory_vector, data_vector, basinscore_method=basinscore_method):
        # OPTION 1 -- is cell in basin yes/no
        # OPTION 2 -- is cell in basin - some scalar value based on ... ?
        # OPTION 3 -- based on compare if data vec in set of basin states (from aux fn)
        hd = hamming(memory_vector, data_vector)
        if data_vector in basin_k[hd]:
            print "data_vector in basin_k[hd]"
            return 1.0
        else:
            return 0.0

    # 1 is build J_ij from Xi
    _, a_inv_arr = memory_corr_matrix_and_inv()
    intxn_matrix = interaction_matrix(memory_array, a_inv_arr, "projection")

    # 2 is score each cell in each cluster based on method
    score_dict = {k: 0 for k in xrange(num_clusters)}
    for k in xrange(num_clusters):
        binary_cluster_data = binarized_cluster_dict[k]
        memory_k = memory_array[:,k]
        basin_k = build_basin_states(intxn_matrix, memory_k)

        for cell_data in binary_cluster_data.T:  # TODO make sure his gives columns (len is N)
            print len(cell_data), num_genes, cell_data.shape
            score_dict[k] += basin_score_pairwise(basin_k, memory_k, cell_data, basinscore_method=basinscore_method)
        print score_dict
    return score_dict


def plot_basins_scores(score_dict):
    """
    Args:
        - score_dict: {k: M x 1 array for k in 0 ... K-1 (i.e. cluster index)}
    Returns:
        - plot axis
    """
    # 1 is build J_ij from Xi
    # 2 is score each cell in each cluster based on method
    # 3 id store scores in score_dict and return
    num_clusters = np.max(score_dict.keys())
    x_axis = range(num_clusters)
    y_axis = [score_dict[k] for k in xrange(num_clusters)]
    plt.bar(x_axis, y_axis)
    plt.title('Basin scores for each cluster')
    plt.xlabel('cluster idx')
    plt.ylabel('basin score')
    return plt.gca()


if __name__ == '__main__':
    # settings
    dataname = None
    labelname = None
    datadir = DATADIR
    datapath = DATADIR + os.sep + dataname
    labelpath = DATADIR + os.sep + labelname
    # methods
    verbose = True
    binarize_method = "default"  # in ['columnwise_midpt', 'clusterwise_midpt', 'global_midpt', ????more????]
    memory_method = "default"
    basinscore_method = "default"
    # analysis
    cluster_dict, metadata = read_exptdata_from_files(dataname, labelname, datadir=datadir, verbose=verbose)
    binarized_cluster_dict = binarize_cluster_dict(cluster_dict, metadata, binarize_method=binarize_method)
    memory_array = binary_cluster_dict_to_memories(binarized_cluster_dict, metadata, memory_method=memory_method)
    basin_scores = get_basins_scores(memory_array, binarized_cluster_dict, metadata, basinscore_method=basinscore_method)
    # plotting
    plot_basins_scores(basin_scores)
