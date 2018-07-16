import numpy as np
import os
import re

from data_settings import DATADIR, RAWDATA_2014MEHTA, RAWDATA_2018SCMCA

"""
Standardize: convert different formats of scRNA expression data into local standard
    - store row labels (genes), column labels (cell ID), and data (expression levels) in one format
    - this format is compressed ".npz" files 
    - save via: save_npz_of_arr_genes_cells(...) calls numpy's savez_compressed(...)
      i.e. np.savez_compressed(npzpath, arr=arr, genes=genes, cells=cells) -- all as numpy arrays
    - load via: load_npz_of_arr_genes_cells(...) calls numpy's load(...)
    - loaded npz acts similar to dictionary
        loaded = np.load(npzpath)
        arr = loaded['arr']
        genes = loaded['genes']
        cells = loaded['cells']
"""
# TODO pass metadata to all functions?
# TODO test and optimize read_exptdata_from_files
# TODO unit tests for pruning and clustering (e.g. augment SI_mehta_zscore file to 10x10 size or so)


def read_datafile_simple(datapath, verbose=True, txt=False):
    """
    Loads file at datapath, which is either a txt file or npy file
    """
    if txt:
        assert datapath[-4:] == '.txt'
        arr = np.loadtxt(datapath)
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


def load_singlecell_data(zscore_datafile=RAWDATA_2014MEHTA, savenpz='mems_genes_clusters_raw_compressed'):
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


def save_npz_of_arr_genes_cells(npzpath, arr, genes, cells):
    np.savez_compressed(npzpath, arr=arr, genes=genes, cells=cells)
    return


def binarize_data(xi):
    return 1.0 * np.where(xi > 0, 1, -1)  # mult by 1.0 to cast as float


def parse_exptdata(states_raw, gene_labels, verbose=True):
    """
    Args:
        - states_raw: stores array of state data and cluster labels for each cell state (column)
        - gene_labels: stores row names i.e. gene or PCA labels (expect list or numpy array)
    Notes: data format may change with time
        - convention is first row stores cluster index, from 0 to np.max(row 0) == K - 1
        - future convention may be to store unique integer ID for each column corresponding to earlier in pipeline
        - maybe also extract and pass metadata_dict info (i.e. K, N, M, filename information on pipeline)
    Returns:
        - cluster_dict: {cluster_idx: N x M array of raw cell states in the cluster (i.e. not binarized)}
        - metadata: dict, mainly stores N x 1 array of 'gene_labels' for each row
    """
    if type(gene_labels) is np.ndarray:
        gene_labels = gene_labels.tolist()
    else:
        assert type(gene_labels) is list

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
    datadir = DATADIR
    # run flags
    flag_load_simple = False
    flag_load_compressed_npz = False
    flag_standardize_2018scMCA = False
    flag_standardize_2014mehta = False

    # simple data load
    if flag_load_simple:
        datapath = "insert path"
        is_txtfile = False
        arr = read_datafile_simple(datapath, verbose=True, txt=is_txtfile)

    if flag_load_compressed_npz:
        compressed_file = "insert path"
        arr, genes, cells = load_npz_of_arr_genes_cells(compressed_file)

    if flag_standardize_2018scMCA:
        datapath = RAWDATA_2018SCMCA
        arr, genes, cells = read_datafile_manual(datapath, verbose=True)

    if flag_standardize_2014mehta:
        # part 1: load their zscore textfile, save in standard npz format
        expression_data, genes, celltypes = load_singlecell_data(zscore_datafile=RAWDATA_2014MEHTA,
                                                                 savenpz='mehta_mems_genes_types_zscore_compressed.npz')
