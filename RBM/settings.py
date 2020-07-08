import os


DIR_DATA = 'data'
DIR_MODELS = 'models'
DIR_OUTPUT = 'output'
DEFAULT_HOPFIELD = DIR_MODELS + os.sep + 'saved' + os.sep + 'hopfield_mnist_10.npz'

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
K_PATTERN_DIV = 1
BETA = 8.0
HRBM_LOGREG_STEPS = 1
HRBM_MANUAL_MAXSTEPS = 10
VISIBLE_FIELD = False
