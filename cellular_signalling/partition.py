from constants import *
import numpy as np

# compute partition function
Z_receptors = [1.0 for _ in xrange(NUM_TYPE_RECEPTOR)]
Z_tot = 1.0
for j in xrange(NUM_TYPE_RECEPTOR):
    Z_receptors[j] = Z_receptors[j] + np.sum([(eps[i][j]-mu[i])/KBT for i in xrange(NUM_TYPE_LIGAND)])
    Z_tot = Z_tot * ( Z_receptors[j]**(m[j]) )

# compute mean binding
r_mean = [0.0 for _ in xrange(NUM_TYPE_RECEPTOR)]
for j in xrange(NUM_TYPE_RECEPTOR):
    r_mean[j] = sum([k * (Z_receptors[j] - 1)**k / Z_receptors[j]**m[j] for k in xrange(m[j])])

print r_mean  # TEST



