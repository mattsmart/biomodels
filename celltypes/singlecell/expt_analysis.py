import matplotlib.pyplot as plt
import numpy as np
import os

from expt_data_handling import parse_exptdata, load_npz_of_arr_genes_cells
from singlecell_constants import DATADIR
from singlecell_functions import hamiltonian, hamming
from singlecell_simsetup import memory_corr_matrix_and_inv, interaction_matrix

# TODO pass metadata to all functions?
# TODO test and optimize build_basin_states
# TODO build remaining functions
# TODO build unit tests pycharm properly


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
    for k in xrange(num_clusters):
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


def is_energy_increase(intxn_matrix, data_vec_a, data_vec_b):
    # state b in basin if the energy from a to b increases AND a is in basin
    energy_a = hamiltonian(data_vec_a, intxn_matrix=intxn_matrix)
    energy_b = hamiltonian(data_vec_b, intxn_matrix=intxn_matrix)
    if energy_b > energy_a:
        return True
    else:
        return False


def build_basin_states(intxn_matrix, memory_vec,
                       recurse_dist_d=0, recurse_basin_set=None, recurse_state=None,
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

    if recurse_basin_set is None:
        memory_vec_copy = np.array(memory_vec[:])
        recurse_basin_set = {d: set() for d in xrange(num_genes + 1)}
        recurse_basin_set[0].add(tuple(memory_vec_copy))
        recurse_state = memory_vec_copy
        sites_flipped_already = []
        recurse_dist_d = 1

    #size_basin_at_dist_d = len(recurse_basin_set[recurse_dist_d])    # number of states with hamming dist = d in the basin

    for site_idx in [val for val in xrange(num_genes) if val not in sites_flipped_already]:
        recurse_state_flipped = np.array(recurse_state[:])
        recurse_state_flipped[site_idx] = -1 * recurse_state[site_idx]
        if is_energy_increase(intxn_matrix, recurse_state, recurse_state_flipped):
            recurse_basin_set[recurse_dist_d].add(tuple(recurse_state_flipped))
            recurse_sites_flipped_already = sites_flipped_already[:]
            recurse_sites_flipped_already.append(site_idx)
            build_basin_states(intxn_matrix, memory_vec,
                               recurse_dist_d=recurse_dist_d + 1,
                               recurse_basin_set=recurse_basin_set,
                               recurse_state=recurse_state_flipped,
                               sites_flipped_already=recurse_sites_flipped_already)
        else:
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
        if tuple(data_vector) in basin_k[hd]:
            print "data_vector in basin_k[hd]"
            return 1.0
        else:
            print "data_vector NOT in basin_k[hd]"
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
    # run flags
    datadir = DATADIR + os.sep + "scMCA"
    flag_load_compressed = True
    flag_basinscore = True

    # options
    verbose = True
    binarize_method = "default"  # in ['columnwise_midpt', 'clusterwise_midpt', 'global_midpt', ????more????]
    memory_method = "default"
    basinscore_method = "default"

    if flag_load_compressed:
        npzpath = datadir + os.sep + 'arr_genes_cells_withcluster_compressed.npz'
        arr, genes, cells = load_npz_of_arr_genes_cells(npzpath)
        print arr.shape, genes.shape, cells.shape

    if flag_basinscore:
        # analysis
        cluster_dict, metadata = parse_exptdata(arr, genes, verbose=verbose)
        binarized_cluster_dict = binarize_cluster_dict(cluster_dict, metadata, binarize_method=binarize_method)
        memory_array = binary_cluster_dict_to_memories(binarized_cluster_dict, metadata, memory_method=memory_method)
        basin_scores = get_basins_scores(memory_array, binarized_cluster_dict, metadata, basinscore_method=basinscore_method)

        # plotting
        plot_basins_scores(basin_scores)
