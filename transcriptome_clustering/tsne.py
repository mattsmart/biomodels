import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.manifold import TSNE
"""
Journal Ref:
"Single-cell RNA-seq reveals new types of human blood dendritic cells, monocytes, and progenitors" 
Villani et al., 2017, Science

Source Documentation:
t-distributed Stochastic Neighbor Embedding
http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

Note from docs:
It is highly recommended to use another dimensionality reduction
method (e.g. PCA for dense data or TruncatedSVD for sparse data)
to reduce the number of dimensions to a reasonable amount (e.g. 50)
if the number of features is very high. This will suppress some
noise and speed up the computation of pairwise distances between
samples.

Raw data files found at or around:
https://www.ncbi.nlm.nih.gov/gds/?term=GSE94820[Accession]

Goal:
Recreate figures like S2
"""


def load_data_from_text(raw_folder, raw_filename):
    raw_filepath = raw_folder + os.sep + raw_file
    with open(raw_filepath) as f:
        idx = 0
        for idx, line in enumerate(f):
            if idx == 0:
                num_col = len(line.split('\t'))
            pass
        num_row = idx
        labels_genes = [0]*(num_row)
        data_raw = np.zeros((num_row, num_col))  # minus 1 bc of row labels (genes), col labels (cells)
        f.seek(0)  # need to point back to tart of file to start reading in lines
        for idx, line in enumerate(f):
            line_cleaned = line.replace('\n', '')
            line_split = line_cleaned.split('\t')
            if idx == 0:
                labels_cells = line_split
            else:
                labels_genes[idx-1] = line_split[0]
                for col, val_as_str in enumerate(line_split[1:]):
                    data_raw[idx-1, col] = float(val_as_str)
        return data_raw, labels_cells, labels_genes


def data_random_prune(data, names_cols, names_rows, frac_col_remove, frac_row_remove):
    # Note: could also sort in here after sampling
    assert 0 <= frac_col_remove <= 1
    assert 0 <= frac_row_remove <= 1
    num_cols_to_keep = len(names_cols) - int(frac_col_remove*len(names_cols))
    num_rows_to_keep = len(names_rows) - int(frac_row_remove*len(names_rows))
    cols_to_keep = random.sample(range(len(names_cols)), num_cols_to_keep)
    rows_to_keep = random.sample(range(len(names_rows)), num_rows_to_keep)
    pruned_data = np.zeros((num_rows_to_keep, num_cols_to_keep))
    pruned_names_cols = [0]*num_cols_to_keep
    pruned_names_rows = [0]*num_rows_to_keep
    for idx_row, row in enumerate(rows_to_keep):
        pruned_names_rows[idx_row] = names_rows[row]
        for idx_col, col in enumerate(cols_to_keep):
            pruned_names_cols[idx_col] = names_cols[col]
            pruned_data[idx_row, idx_col] = data[row, col]
    return pruned_data, pruned_names_cols, pruned_names_rows


# LOAD DATA
raw_folder = "rawdata"
raw_file = "GSE94820_raw.expMatrix_DCnMono.discovery.set.submission.txt"
frac_col_remove = 0.90
frac_row_remove = 0.85
print "Reading raw data..."
data_raw, labels_cells, labels_genes = load_data_from_text(raw_folder, raw_file)
print "Raw data successfully loaded: size %d by %d" % (len(labels_genes), len(labels_cells))
print "Pruning data..."
data_pruned, labels_cells_pruned, labels_genes_pruned = data_random_prune(data_raw, labels_cells, labels_genes,
                                                                          frac_col_remove, frac_row_remove)
print "Raw data pruned: new size %d by %d" % (len(labels_genes_pruned), len(labels_cells_pruned))

# CLEAN THE RAW DATA AS PRESCRIBED IN (Villani 2017)
#TODO
# CREATE TEST DATA WITH FAKE ROWS
#TODO

# PERFORM CLUSTERING
N_COMPONENTS = 2
VERBOSE = 2
tsne_obj = TSNE(n_components=N_COMPONENTS, verbose=VERBOSE)
print "Performing t-SNE on data..."
data_projected = tsne_obj.fit_transform(data_pruned)  # TODO: does data need to be transposed first?
print "Done t-SNE"
print type(data_projected), np.shape(data_projected), len(data_projected)  #size is numrows x 2
tsne_x = data_projected[:,0]
tsne_y = data_projected[:,1]
plt.plot(tsne_x,tsne_y)
plt.show()

# ANALYZE CLUSTERING
#TODO
