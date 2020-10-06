import numpy as np
import os
from sklearn import svm
from sklearn.linear_model import LogisticRegression


DIR_DATA = 'data'
DIR_MODELS = 'models'
DIR_OUTPUT = 'output'
DIR_CLASSIFY = DIR_OUTPUT + os.sep + 'classify'
DIR_TRAINING = DIR_OUTPUT + os.sep + 'training'
DEFAULT_HOPFIELD = DIR_MODELS + os.sep + 'saved' + os.sep + 'hopfield_mnist_10.npz'

for dir in [DIR_DATA, DIR_MODELS, DIR_CLASSIFY, DIR_TRAINING]:
    if not os.path.exists(dir):
        os.makedirs(dir)

CPU_THREADS = 8
BATCH_SIZE = 4

SYNTHETIC_DIM = 8
SYNTHETIC_SAMPLES = 10000
SYNTHETIC_NOISE_VALID = ['symmetric']
SYNTHETIC_SAMPLING_VALID = ['balanced']
SYNTHETIC_DATASPLIT = ['balanced']

DATA_CHOICE = 'mnist'

# HYPERPARAMETERS
# For threshold 0.4 (no pre-binarizing)
#   2.0 -> 65.5%, 5.0 -> 68.6%, 20.0 -> 67.5%, 200.0 -> 65.6%
# Best accuracies so far:
#    71% for ON threshold 0.3, pattern threshold 0.0, and beta 5.0
#    72% for ON threshold 0.2, pattern threshold 0.0, and beta 5.0
#    74% for ON threshold 0.1, pattern threshold 0.0, and beta 5.0
#    74.4% for ON threshold 0.05, pattern threshold 0.0, and beta 5.0
#    74.9% for ON threshold 0.02, pattern threshold 0.0, and beta 5.0
#    75% for ON threshold 0.01, pattern threshold 0.0, and beta 5.0
#    75% for ON threshold 0.001, pattern threshold 0.1, and beta 5.0
#    75% for ON threshold 0.01, pattern threshold 0.0, and beta 8.0 <---- main
MNIST_BINARIZATION_CUTOFF = 0.01
PATTERN_THRESHOLD = 0.0
K_PATTERN_DIV = 5
BETA = 2.0
GAUSSIAN_STDEV = np.sqrt(1.0/BETA)
VISIBLE_FIELD = False

HRBM_MANUAL_MAXSTEPS = 5
HRBM_CLASSIFIER_STEPS = 1
USE_BETA_SCHEDULER = False

USE_SVM = False
if USE_SVM:
    # as in https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html;
    # FIRST USED:      gamma=0.001, C=1.0       -> p=50: 0.948 (10k), 0.961 (60k)
    # ok/maybe better: gamma=1e-4, C=1.0        -> p=50: 0.955 (10k), 0.962 (60k)
    # ok/maybe better: gamma=1e-4, C=6.0        -> p=50:              0.964 (60k)
    # ok/maybe better: gamma=1.04*1e-4, C=6.0   -> p=50:              0.965 (60k) (best)
    # ok/maybe better: gamma=1e-5, C=10.0       -> p=50: 0.953 (10k), 0.960 (60k)
    # try 'auto' or 'scale'
    #CLASSIFIER = svm.SVC(gamma=1.04*1e-4, C=6)
    CLASSIFIER = svm.SVC(gamma=0.001, C=1.0)
    # TODO try revert to 0.21.? from 0.23 to see if baseline improves
else:
    # FIRST USED:      C=1e5, multi_class='multinomial', penalty='l1', solver='saga', tol=0.1
    # New:             C=1e5, multi_class='multinomial', penalty='l1', solver='saga', tol=0.001  decreasing C is OK
    CLASSIFIER = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l1', solver='saga', tol=0.001, max_iter=100)  # Def: 100 max iter
