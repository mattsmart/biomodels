import matplotlib.pyplot as plt
import numpy as np

# TODO pass metadata to all functions?
# TODO test and optimize read_exptdata_from_files
# TODO build remaining functions
# TODO change code file structure with expt data IO folder in constants or expt_constants?


def read_exptdata_from_files(datapath, labelpath, verbose=True):
    """
    Args:
        - datapath stores array of state data and cluster labels for each cell state (column)
        - labelpath stores row names i..e gene or PCA labels
    Notes: data format may change with time
        - convention is first row stores cluster index, from 0 to np.max(row 0) == K - 1
        - future convention may be to store unique integer ID for each column corresponding to earlier in  pipeline
        - maybe also extract and pass metadata_dict info (i.e. K, N, M, filename information on pipeline)
    Returns:
        - cluster_dict: {cluster_idx: N x M array of raw cell states in the cluster (i.e. not binarized)}
        - metadata_dict: mainly stores N x 1 array of 'gene_labels' for each row
    """
    # load data
    states_raw = np.loadtxt(datapath, delimiter=",")
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
    gene_labels = np.loadtxt(labelpath, delimiter=",", dtype=float)
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
    metadata['datapath'] = datapath    # TODO parse for useful info, maybe need dir and filename as args not path
    metadata['labelpath'] = labelpath  # TODO parse for useful info, maybe need dir and filename as args not path

    return cluster_dict, metadata



def binarize_cluster_dict(cluster_dict, binarize_method="default"):
    """
    Args:
        - cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - binarize_method: options for different binarization methods
    Returns:
        - binarized_bluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
    """
    return 0


def cluster_dict_to_memories(cluster_dict, metadata, memory_method="default"):
    """
    Args:
        - cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - memory_method: options for different memory processing algos
    Returns:
        - memory_array: i.e. xi matrix, will be N x K (one memory from each cluster)
    """
    num_genes = metadata['num_genes']
    num_cells = metadata['num_cells']
    num_clusters = metadata['num_clusters']
    memory_array = 0
    return memory_array


def get_basins_scores(memory_array, binarized_bluster_dict, basinscore_method="default"):
    """
    Args:
        - binarized_bluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - basinscore_method: options for different basin scoring algos
    Returns:
        - score_dict: {k: M x 1 array for k in 0 ... K-1 (i.e. cluster index)}
    """
    # 1 is build J_ij from Xi
    # 2 is score each cell in each cluster based on method
    # 3 id store scores in score_dict and return
    score_dict = 0
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
