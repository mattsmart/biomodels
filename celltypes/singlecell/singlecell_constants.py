import os


# IO
ZSCORE_DATAFILE = "mehta_zscore_table" + os.sep + "mehta_zscore_table.txt"
RUNS_FOLDER = "runs" + os.sep             # store timestamped runs here

# SINGLECELL SIMULATION CONSTANTS
BETA = 2.2                                # value used in Mehta 2014
METHOD = "projection"                     # 'projection' or 'hopfield'
FLAG_BOOL = True                          # use binarized states (up/down vs continuous)
NUM_STEPS = 100                           # number of timesteps to loop for
