import numpy as np
from sklearn.manifold import TSNE
"""
Source Documentation:
http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

It is highly recommended to use another dimensionality reduction
method (e.g. PCA for dense data or TruncatedSVD for sparse data)
to reduce the number of dimensions to a reasonable amount (e.g. 50)
if the number of features is very high. This will suppress some
noise and speed up the computation of pairwise distances between
samples.
"""

# LOAD DATA
data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# CREATE TEST DATA
# TODO

# PERFORM CLUSTERING
tsne_obj = TSNE(n_components=TODO)
data_projected = tsne_obj.fit_transform(data)

# ANALYZE CLUSTERING