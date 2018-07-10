import numpy as np
import os
import re

from singlecell_constants import DATADIR, MEHTA_ZSCORE_DATAFILE_PATH

# TODO pass metadata to all functions?
# TODO test and optimize read_exptdata_from_files


def read_datafile_simple(datapath, verbose=True, txt=False):
    if txt:
        assert datapath[-4:] == '.txt'
        arr = np.loadtxt(datapath, skiprows=1, usecols=(1,3,4))
    else:
        arr = np.load(datapath)
    if verbose:
        print "Loaded dim %s data at %s" % (arr.shape, datapath)
    return arr


def read_datafile_manual(datapath, verbose=True, save_as_sep_npy=False):
    """
    Datafile form is N+1 x M+1 array with column and row labels (NxM expression matrix)
    Function created for loading mouse cell atlas data formats
    http://bis.zju.edu.cn/MCA/contact.html (DGE matrices)
    """
    assert datapath[-4:] == '.txt'
    with open(datapath) as f:
        count = 0
        # do first pass to get gene count and cell names
        for idx, line in enumerate(f):
            if idx == 0:
                line = line.rstrip()
                line = line.split('\t')
                cell_names = [a.strip('"') for a in line]
            else:
                count += 1
        arr = np.zeros((count, len(cell_names)), dtype=np.int16)  # max size ~ 33k (unsigned twice that)
        gene_names = [0] * count
        print "data size will be (genes x cells):", arr.shape
        # second pass to get gene names and array data
        f.seek(0)
        for idx, line in enumerate(f):
            if idx > 0:
                if idx % 1000 == 0:
                    print idx
                line = line.rstrip()
                line = line.split('\t')
                gene_names[idx-1] = line[0].strip('"')
                arr[idx-1, :] = [int(val) for val in line[1:]]
    if verbose:
        print "Loaded dim %s data at %s" % (arr.shape, datapath)
        print "Max val in arr:", np.max(arr)
        print "Size of arr in memory (bytes)", arr.nbytes

    datadir = os.path.abspath(os.path.join(datapath, os.pardir))
    if save_as_sep_npy:
        np.save(datadir + os.sep + "raw_arr.npy", arr)
        np.save(datadir + os.sep + "raw_genes.npy", np.array(gene_names))
        np.save(datadir + os.sep + "raw_cells.npy", np.array(cell_names))
    else:
        genes = np.array(gene_names)
        cells = np.array(cell_names)
        np.savez_compressed(datadir + os.sep + "arr_genes_cells_raw_compressed.npz", arr=arr, genes=genes, cells=cells)
    return arr, gene_names, cell_names


def load_singlecell_data(zscore_datafile=MEHTA_ZSCORE_DATAFILE_PATH, savenpz='mems_genes_clusters_raw_compressed'):
    """
    Returns list of cell types (size p), list of TFs (size N), and xi array where xi_ij is ith TF value in cell type j
    Note the Mehta SI file has odd formatting (use regex to parse); array text file is read in as single line:
    http://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1003734.s005&type=supplementary
    """
    gene_labels = []
    with open(zscore_datafile) as f:
        origline = f.readline()
        filelines = origline.split('\r')
        for idx_row, row in enumerate(filelines):
            row_split = re.split(r'\t', row)
            if idx_row == 0:  # celltypes line, first row
                celltype_labels = row_split[1:]
            else:
                gene_labels.append(row_split[0])
    # reloop to get data without excessive append calls
    expression_data = np.zeros((len(gene_labels), len(celltype_labels)))
    with open(zscore_datafile) as f:
        origline = f.readline()
        filelines_dataonly = origline.split('\r')[1:]
        for idx_row, row in enumerate(filelines_dataonly):
            row_split_dataonly = re.split(r'\t', row)[1:]
            expression_data[idx_row,:] = [float(val) for val in row_split_dataonly]
    if savenpz is not None:
        datadir = os.path.abspath(os.path.join(zscore_datafile, os.pardir))
        npzpath = datadir + os.sep + savenpz
        save_npz_of_arr_genes_cells(npzpath, expression_data, gene_labels, celltype_labels)
    return expression_data, gene_labels, celltype_labels


def load_npz_of_arr_genes_cells(npzpath, verbose=True):
    """
    Can also use for memory array, gene labels, and cell cluster names!
    """
    if verbose:
        print "loading npz of arr genes cells at", npzpath, "..."
    loaded = np.load(npzpath)
    arr = loaded['arr']
    genes = loaded['genes']
    cells = loaded['cells']
    if verbose:
        print "loaded arr, genes, cells:", arr.shape, genes.shape, cells.shape
    return arr, genes, cells


def binarize_data(xi):
    return 1.0 * np.where(xi > 0, 1, -1)  # mult by 1.0 to cast as float


def save_npz_of_arr_genes_cells(npzpath, arr, genes, cells):
    np.savez_compressed(npzpath, arr=arr, genes=genes, cells=cells)
    return


def attach_cluster_id_arr_manual(npzpath, clusterpath, save=True, one_indexed=True):
    """
    one_indexed: if true, assume cluster index starts at 1 (as in scMCA)
    """
    arr, genes, cells = load_npz_of_arr_genes_cells(npzpath)
    # generate cell_to_cluster_idx mapping
    cluster_info = {}
    # expected format of csv file is "cell name, cluster idx, tissue origin"
    with open(clusterpath) as f:
        for idx, line in enumerate(f):
            line = line.rstrip()
            line = line.split(',')
            cluster_info[line[0]] = int(line[1])
    # adjust genes and arr contents
    arr = np.insert(arr, 0, 0, axis=0)
    genes = np.insert(genes, 0, 'cluster_id')  # TODO should have global constant for this mock gene label
    if one_indexed:
        for idx in xrange(len(cells)):
            arr[0,idx] = cluster_info[cells[idx]] - 1
    else:
        for idx in xrange(len(cells)):
            arr[0,idx] = cluster_info[cells[idx]]
    # save and return data
    if save:
        print "saving cluster-appended arrays..."
        np.savez_compressed(datadir + os.sep + "arr_genes_cells_withcluster_compressed.npz", arr=arr, genes=genes, cells=cells)
    return arr, genes, cells


def prune_boring_rows(npzpath, save=True):
    """
    Delete rows from array and corresponding genes that are self-duplicates
    NOTE: very similar to reduce_gene_set(xi, gene_labels)
    """

    arr, genes, cells = load_npz_of_arr_genes_cells(npzpath)
    # collect rows to delete
    rows_duplicates = np.all(arr.T == arr.T[0,:], axis=0)
    rows_to_delete = np.array([idx for idx, val in enumerate(rows_duplicates) if val])
    # note pruned rows
    print "number of self-duplicate rows:", len(rows_to_delete)
    print rows_to_delete
    print rows_to_delete[0:10]
    # adjust genes and arr contents
    print "Orig shape arr, genes, cells:", arr.shape, genes.shape, cells.shape
    arr = np.delete(arr, rows_to_delete, axis=0)
    genes = np.delete(genes, rows_to_delete)  # TODO should have global constant for this mock gene label
    print "New shape arr, genes, cells:", arr.shape, genes.shape, cells.shape  # TODO not operating as expected
    # save and return data
    if save:
        print "saving pruned arrays..."
        datadir = os.path.abspath(os.path.join(npzpath, os.pardir))
        base = os.path.basename(npzpath)
        basestr = os.path.splitext(base)[0]
        savestr = basestr + '_pruned.npz'
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


def load_cluster_labels(clusterpath, one_indexed=True):
    """
    one_indexed: if true, assume cluster index starts at 1 (as in scMCA)
    """
    cluster_labels = {}
    if one_indexed:
        dec = 1
    else:
        dec = 0
    # expected format of csv file is "cluster number, name"
    with open(clusterpath) as f:
        for idx, line in enumerate(f):
            line = line.rstrip()
            line = line.split(',')
            print line
            cluster_labels[int(line[0])-dec] = line[1]
    return cluster_labels


def parse_exptdata(states_raw, gene_labels, verbose=True):
    """
    Args:
        - states_raw: stores array of state data and cluster labels for each cell state (column)
        - gene_labels: stores row names i.e. gene or PCA labels
    Notes: data format may change with time
        - convention is first row stores cluster index, from 0 to np.max(row 0) == K - 1
        - future convention may be to store unique integer ID for each column corresponding to earlier in pipeline
        - maybe also extract and pass metadata_dict info (i.e. K, N, M, filename information on pipeline)
    Returns:
        - cluster_dict: {cluster_idx: N x M array of raw cell states in the cluster (i.e. not binarized)}
        - metadata: dict, mainly stores N x 1 array of 'gene_labels' for each row
    """
    states_row0 = states_raw[0, :]
    states_truncated = states_raw[1:, :]
    num_genes, num_cells = states_truncated.shape  # aka N, M
    num_clusters = np.max(states_raw[0, :]) + 1
    if verbose:
        print "raw data dimension: %d x %d" % (states_raw.shape)
        print "cleaned data dimension: %d x %d" % (states_truncated.shape)
        print "num_clusters is %d" % num_clusters

    # process gene labels
    assert len(gene_labels) == num_genes or len(gene_labels) == num_genes + 1
    if len(gene_labels) == num_genes + 1:
        assert 'cluster' in gene_labels[0]
        gene_labels = gene_labels[1:]

    # prep cluster_dict
    cluster_dict = {}
    cluster_indices = {k: [] for k in xrange(num_clusters)}
    # TODO optimize this chunk if needed
    for cell_idx in xrange(num_cells):
        cluster_idx = states_row0[cell_idx]
        cluster_indices[cluster_idx].append(cell_idx)

    # build cluster dict
    if verbose:
        print "cluster_indices collected; building cluster arrays..."
    for k in xrange(num_clusters):
        print k
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
    return cluster_dict, metadata


if __name__ == '__main__':
    # run flags
    datadir = DATADIR + os.sep + "2018_scMCA"
    flag_load_simple = False
    flag_process_manual = False
    flag_load_sep_npy = False
    flag_load_compressed_npz = False
    flag_attach_clusters_resave = False
    flag_prune_boring_rows = False
    flag_process_data_mehta = False

    # simple data load
    if flag_load_simple:
        datapath = datadir + os.sep + "SI_Figure2-batch-removed.txt"
        arr = read_datafile_simple(datapath, verbose=True, txt=True)

    if flag_process_manual:
        datapath = datadir + os.sep + "SI_Figure2-batch-removed.txt"
        arr, genes, cells = read_datafile_manual(datapath, verbose=True)

    if flag_load_sep_npy:
        arrpath = datadir + os.sep + "raw_arr.npy"
        genespath = datadir + os.sep + "raw_genes.npy"
        cellspath = datadir + os.sep + "raw_cells.npy"
        arr = read_datafile_simple(arrpath, verbose=True)
        genes = read_datafile_simple(genespath, verbose=True)
        cells = read_datafile_simple(cellspath, verbose=True)

    if flag_load_compressed_npz:
        compressed_file = datadir + os.sep + "arr_genes_cells_raw_compressed.npz"
        arr, genes, cells = load_npz_of_arr_genes_cells(compressed_file)

    if flag_attach_clusters_resave:
        compressed_file = datadir + os.sep + "arr_genes_cells_raw_compressed.npz"
        clusterpath = datadir + os.sep + "SI_cells_to_clusters.csv"
        arr, genes, cells = attach_cluster_id_arr_manual(compressed_file, clusterpath, save=True)

    if flag_prune_boring_rows:
        compressed_file = datadir + os.sep + "arr_genes_cells_raw_compressed.npz"
        prune_boring_rows(compressed_file)

    if flag_process_data_mehta:
        # part 1: load their zscore textfile, save in standard npz format
        expression_data, genes, celltypes = load_singlecell_data(zscore_datafile=MEHTA_ZSCORE_DATAFILE_PATH,
                                                                 savenpz='mehta_mems_genes_clusters_zscore_compressed.npz')
        # part 2: load npz, binarize, save
        npzpath = DATADIR + os.sep + "2014_mehta" + os.sep + 'mehta_mems_genes_clusters_zscore_compressed.npz'
        expression_data, genes, celltypes = load_npz_of_arr_genes_cells(npzpath, verbose=True)
        xi = binarize_data(expression_data)
        compressed_boolean = datadir + os.sep + "mehta_mems_genes_clusters_boolean_compressed.npz"
        save_npz_of_arr_genes_cells(compressed_boolean, xi, genes, celltypes)
        # part 3: load npz, prune, save
        rows_to_delete, xi, genes, celltypes = prune_boring_rows(compressed_boolean)
