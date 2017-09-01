import matplotlib.pyplot as plt
import numpy as np
import os
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

# LOAD DATA
raw_folder = "rawdata"
raw_file = "GSE94820_raw.expMatrix_DCnMono.discovery.set.submission.txt"
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

    #print labels_genes[0:2], labels_genes[-2:]
    #print raw_data[0][0:5], raw_data[0][-5:]
    #print raw_data[-1][0:8], raw_data[-1][-5:]
    print "Raw data successfully loaded"

# CLEAN THE RAW DATA AS PRESCRIBED IN (Villani 2017)


# CREATE TEST DATA
# TODO

# PERFORM CLUSTERING
N_COMPONENTS = 2
VERBOSE = 1
tsne_obj = TSNE(n_components=N_COMPONENTS, verbose=VERBOSE)
print "Performing t-SNE on data"
data_projected = tsne_obj.fit_transform(data_raw)
print "Done"
print type(data_projected), np.shape(data_projected), len(data_projected)
#plt.plot(x,y)
#plt.show()

# ANALYZE CLUSTERING
#TODO
