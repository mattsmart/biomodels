import os
import sys

CELLTYPES = os.path.dirname(os.path.dirname(__file__))
sys.path.append(CELLTYPES)
print "Appended to sys path", CELLTYPES  # TODO can maybe move this too simetup fn call and call once somewhere else...

# IO
SINGLECELL = CELLTYPES + os.sep + "singlecell"
DATADIR = CELLTYPES + os.sep + "input"
MEMORIESDIR = DATADIR + os.sep + "memories"
MEMS_MEHTA = MEMORIESDIR + os.sep + "2014_mehta_mems_genes_types_boolean_compressed_pruned.npz"
MEMS_SCMCA = MEMORIESDIR + os.sep + "2018_scmca_mems_genes_types_boolean_compressed_pruned.npz"
RUNS_FOLDER = SINGLECELL + os.sep + "runs"                      # store timestamped runs here
SETTINGS_FILE = "run_info.txt"

# SINGLECELL SIMULATION CONSTANTS
BETA = 2.2                                # value used in Mehta 2014 (low temperature)
EXT_FIELD_STRENGTH = 0.30                 # relative strength of exosome local field effect
APP_FIELD_STRENGTH = 1.0                  # relative strength of artificial applied fields

FLAG_PRUNE_INTXN_MATRIX = False           # flag for non-eq dilution of the symmetric J
J_RANDOM_DELETE_RATIO = 0.2               # this ratio of elements randomly pruned from J

DEFAULT_MEMORIES_NPZPATH = MEMS_MEHTA     # choose which memories to embed
METHOD = "projection"                     # 'projection' or 'hopfield'
FLAG_BOOL = True                          # use binarized states (up/down vs continuous)  # TODO unused... remove or adjust

NUM_STEPS = 100                           # number of full TF grid updates in the single cell simulation
BURST_ERROR_PERIOD = 5                    # val 5 = apply every 5 full spin updates (~5000 individual spin updates)

# TODO generalize, this is for 2014_MEHTA only
IPSC_CORE_GENES = ['Sox2', 'Pou5f1', 'Klf4', 'Mycbp']  # "yamanaka" factors to make iPSC (labels for mehta dataset)
IPSC_EXTENDED_GENES = IPSC_CORE_GENES + ['Nanog']
