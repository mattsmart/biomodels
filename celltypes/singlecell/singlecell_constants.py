import os

# IO
ZSCORE_DATAFILE_NAME = "mehta_zscore_table.txt"
ZSCORE_DATAFILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ZSCORE_DATAFILE_NAME))
RUNS_FOLDER = "runs" + os.sep             # store timestamped runs here

# SINGLECELL SIMULATION CONSTANTS
BETA = 2.2                                # value used in Mehta 2014
FIELD_STRENGTH = 0.25                       # relative strength of exosome local field effect
METHOD = "projection"                     # 'projection' or 'hopfield'
FLAG_BOOL = True                          # use binarized states (up/down vs continuous)
FLAG_REMOVE_DUPES = True                  # remove genes that have same binarized state across all memories
NUM_STEPS = 100                           # number of full TF grid updates in the single cell simulation
