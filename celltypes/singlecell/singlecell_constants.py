import os

# IO
DATADIR = "data"
MEMS_MEHTA = DATADIR + os.sep + "2014_mehta" + os.sep + "mehta_mems_genes_types_boolean_compressed_pruned.npz"
MEMS_SCMCA = DATADIR + os.sep + "2018_scMCA" + os.sep + "mems_genes_types_compressed_pruned.npz"
DEFAULT_MEMORIES_NPZPATH = MEMS_SCMCA
MEHTA_ZSCORE_DATAFILE_PATH = DATADIR + os.sep + "2014_mehta" + os.sep + "SI_mehta_zscore_table.txt"
RUNS_FOLDER = "runs" + os.sep             # store timestamped runs here

# SINGLECELL SIMULATION CONSTANTS
BETA = 2.2                                # value used in Mehta 2014
EXT_FIELD_STRENGTH = 0.30                 # relative strength of exosome local field effect
APP_FIELD_STRENGTH = 1.0                  # relative strength of artificial applied fields

FLAG_PRUNE_INTXN_MATRIX = False           # flag for non-eq dilution of the symmetric J
J_RANDOM_DELETE_RATIO = 0.2               # this ratio of elements randomly pruned from J

METHOD = "projection"                     # 'projection' or 'hopfield'
FLAG_BOOL = True                          # use binarized states (up/down vs continuous)
FLAG_REMOVE_DUPES = True                  # remove genes that have same binarized state across all memories

NUM_STEPS = 100                           # number of full TF grid updates in the single cell simulation
BURST_ERROR_PERIOD = 5                    # val 5 = apply every 5 full spin updates (~5000 individual spin updates)

IPSC_CORE_GENES = ['Sox2', 'Pou5f1', 'Klf4', 'Mycbp']  # "yamanaka" factors to make iPSC (labels for mehta dataset)
IPSC_EXTENDED_GENES = IPSC_CORE_GENES + ['Nanog']
