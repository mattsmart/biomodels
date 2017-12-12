import numpy as np
import re
from os import sep


zscore_datafile = "mehta_zscore_table" + sep + "mehta_zscore_table.txt"


def load_singlecell_data(zscore_datafile=zscore_datafile):
    """
    Returns list of cell types (size p), list of TFs (size N), and xi array where xi_ij is ith TF value in cell type j
    Note the Mehta SI file has odd formatting, array text file is read in as single line:
    http://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1003734.s005&type=supplementary
    """
    gene_labels = []     # ???
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
    return expression_data, celltype_labels, gene_labels


def binarize_data(xi):
    return np.where(xi > 0, 1, -1)


def state_write(state, filepath):
    return 0


def state_read(state, filepath):
    return 0


if __name__ == '__main__':
    xi, celltype_labels, gene_labels = load_singlecell_data(zscore_datafile)
    xi_bool = binarize_data(xi)
