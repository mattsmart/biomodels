import matplotlib.pyplot as plt
import numpy as np
import os

from data_standardize import parse_exptdata, load_npz_of_arr_genes_cells
from data_process import binarize_cluster_dict, binary_cluster_dict_to_memories
from singlecell.singlecell_constants import DATADIR
from singlecell.singlecell_functions import hamiltonian, hamming, state_memory_projection_single
from singlecell.singlecell_linalg import memory_corr_matrix_and_inv, interaction_matrix, predictivity_matrix
from singlecell.singlecell_simulate import singlecell_sim

"""
Script to check whether each cluster is representative of the basin of attraction in hopfield/projection model
 - build interaction matrix from reduced memory matrix
 - load reduced cluster data
 - get basin score for each cluster by checking if all cells (in that cluster) stay within that basin dynamically
 - plot scores 
"""

# TODO test and optimize build_basin_states
# TODO build remaining functions + unit tests


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


def basin_projection_timeseries(k, memory_array, intxn_matrix, eta, basin_data_k, plot=True):

    def get_memory_proj_timeseries(state_array, memory_idx):
        num_steps = np.shape(state_array)[1]
        timeseries = np.zeros(num_steps)
        for time_idx in xrange(num_steps):
            timeseries[time_idx] = state_memory_projection_single(state_array, time_idx, memory_idx, eta=eta)
        return timeseries

    TEMP = 1e-2
    num_steps = 100
    analysis_subdir = "basinscores"
    proj_timeseries_array = np.zeros((num_steps, basin_data_k.shape[1]))

    for idx in xrange(basin_data_k.shape[1]):
        init_cond = basin_data_k[:, idx]
        cellstate_array, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = singlecell_sim(
            init_state=init_cond, iterations=num_steps, beta=1/TEMP, xi=memory_array, intxn_matrix=intxn_matrix,
            memory_labels=range(memory_array.shape[1]), gene_labels=range(memory_array.shape[0]),
            flag_write=False, analysis_subdir=analysis_subdir, plot_period=num_steps * 2)
        proj_timeseries_array[:, idx] = get_memory_proj_timeseries(cellstate_array, k)[:]
    if plot:
        plt.plot(xrange(num_steps), proj_timeseries_array, color='blue', linewidth=0.75)
        plt.title('Test memory %d projection for all cluster memeber' % k)
        plt.ylabel('proj on memory %d' % (k))
        plt.xlabel('Time (%d updates, all spins)' % num_steps)
        plt.savefig(DATADIR + os.sep + analysis_subdir + os.sep + 'cluster_%d.png' % k)
    return proj_timeseries_array


def get_basins_scores(memory_array, binarized_cluster_dict, metadata, basinscore_method="default"):
    """
    Args:
        - memory_array: i.e. xi matrix, will be N x K (one memory from each cluster)
        - binarized_cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - metadata: dict, mainly stores N x 1 array of 'gene_labels' for each row
        - basinscore_method: options for different basin scoring algos
                             (one based on crawling the basin exactly, other via low temp dynamics)
    Returns:
        - score_dict: {k: M x 1 array for k in 0 ... K-1 (i.e. cluster index)}
    """
    assert basinscore_method in ['crawler', 'trajectories']
    num_genes = metadata['num_genes']
    num_cells = metadata['num_cells']
    num_clusters = metadata['num_clusters']

    def basin_score_pairwise(basin_k, memory_vector, data_vector):
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
    np.savetxt(DATADIR + os.sep + 'xi.csv', memory_array)
    _, a_inv_arr = memory_corr_matrix_and_inv(memory_array, check_invertible=True)
    eta = predictivity_matrix(memory_array, a_inv_arr)
    intxn_matrix = interaction_matrix(memory_array, a_inv_arr, "projection")

    # 2 is score each cell in each cluster based on method
    score_dict = {k: 0 for k in xrange(num_clusters)}

    if basinscore_method == 'crawler':
        for k in xrange(num_clusters):
            print "Scoring basin for cluster", k
            binary_cluster_data = binarized_cluster_dict[k]
            memory_k = memory_array[:,k]
            basin_k = build_basin_states(intxn_matrix, memory_k)
            for cell_data in binary_cluster_data.T:  # TODO make sure his gives columns (len is N)
                print len(cell_data), num_genes, cell_data.shape
                score_dict[k] += basin_score_pairwise(basin_k, memory_k, cell_data)
            print score_dict
    else:
        assert basinscore_method == 'trajectories'
        for k in xrange(num_clusters):
            init_conds = binarized_cluster_dict[k]
            trajectories = basin_projection_timeseries(k, memory_array, intxn_matrix, eta, init_conds)
            score_dict[k] = np.mean(trajectories[-1,:])
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
    datadir = DATADIR + os.sep + "2018_scMCA"
    flag_basinscore = True

    # options
    verbose = True
    binarize_method = "by_gene"  # either 'by_cluster', 'by_gene'
    memory_method = "default"
    basinscore_method = "trajectories"  # either 'trajectories', 'crawler'

    rawdata_npzpath = datadir + os.sep + 'arr_genes_cells_withcluster_compressed.npz'

    if flag_basinscore:
        # (1) load pruned raw data
        rawpruned_path = datadir + os.sep + 'arr_genes_cells_withcluster_compressed_pruned.npz'
        arr, genes, cells = load_npz_of_arr_genes_cells(rawpruned_path, verbose=True)
        # (2) create pruned cluster dict
        cluster_dict, metadata = parse_exptdata(arr, genes, verbose=verbose)
        # (3) binarize cluster dict
        binarized_cluster_dict = binarize_cluster_dict(cluster_dict, metadata, binarize_method=binarize_method)
        # (4) create memory matrix
        #     - alternative: load memory array from file and assert gene lists same for example as QC check
        memory_array = binary_cluster_dict_to_memories(binarized_cluster_dict, metadata, memory_method=memory_method)
        # (5) basin scores
        basin_scores = get_basins_scores(memory_array, binarized_cluster_dict, metadata,
                                         basinscore_method=basinscore_method)
        # (6) plotting
        plot_basins_scores(basin_scores)
