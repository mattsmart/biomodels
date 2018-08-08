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

# MODEL SPECIFICATION -- TODO print used vars in simsetup dict, write to run_info.txt
DEFAULT_MEMORIES_NPZPATH = MEMS_MEHTA     # choose which memories to embed
METHOD = "projection"                     # 'projection' or 'hopfield'
BETA = 2.2                                # value used in Mehta 2014 (low temperature)
EXT_FIELD_STRENGTH = 0.30                 # relative strength of exosome local field effect
APP_FIELD_STRENGTH = 1.0                  # relative strength of artificial applied fields
FLAG_BOOL = True                          # use binarized states (up/down vs continuous)  # TODO unused remove/adjust
FLAG_PRUNE_INTXN_MATRIX = False           # flag for non-eq dilution of the symmetric J intxn matrix
J_RANDOM_DELETE_RATIO = 0.2               # this ratio of elements randomly pruned from J intxn matrix

# SPIN FLIP DYNAMICS -- TODO print used vars in simsetup dict, write to run_info.txt
NUM_FULL_STEPS = 100                      # number of full TF grid updates in the single cell simulation
METHOD_STEPS = 'async_batch'              # options: 'async_indiv' (select one spin at a time) or 'async_batch'
FLAG_BURST_ERRORS = False                 # forced spin swaps/errors to randomly apply every T full spin updates
BURST_ERROR_PERIOD = 5                    # val 5 = apply every 5 full spin updates (~5000 individual spin updates)

# SPECIFIC APPLIED FIELDS
# TODO generalize, this is for 2014_MEHTA only
IPSC_CORE_GENES = ['Sox2', 'Pou5f1', 'Klf4', 'Mycbp']  # "yamanaka" factors to make iPSC (labels for mehta dataset)
IPSC_CORE_GENES_EFFECTS = {label: 1.0 for label in IPSC_CORE_GENES}           # this ensure all should be ON
IPSC_EXTENDED_GENES = IPSC_CORE_GENES + ['Nanog']
IPSC_EXTENDED_GENES_EFFECTS = {label: 1.0 for label in IPSC_EXTENDED_GENES}   # TODO ensure all should be ON
