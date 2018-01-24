import os

# IO
ZSCORE_DATAFILE_NAME = "mehta_zscore_table.txt"
ZSCORE_DATAFILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ZSCORE_DATAFILE_NAME))
RUNS_FOLDER = "runs" + os.sep             # store timestamped runs here

# SINGLECELL SIMULATION CONSTANTS
BETA = 2.2                                # value used in Mehta 2014
FIELD_STRENGTH = 0.30                     # relative strength of exosome local field effect

FLAG_PRUNE_INTXN_MATRIX = False           # flag for non-eq dilution of the symmetric J
J_RANDOM_DELETE_RATIO = 0.2               # this ratio of elements randomly pruned from J

METHOD = "projection"                     # 'projection' or 'hopfield'
FLAG_BOOL = True                          # use binarized states (up/down vs continuous)
FLAG_REMOVE_DUPES = True                  # remove genes that have same binarized state across all memories

NUM_STEPS = 100                           # number of full TF grid updates in the single cell simulation
BURST_ERROR_PERIOD = 5                    # val 5 = apply every 5 full spin updates (~5000 indiv spin updates)
