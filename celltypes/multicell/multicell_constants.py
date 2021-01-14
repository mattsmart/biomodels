BLOCK_UPDATE_LATTICE = True
GRIDSIZE = 4
SEARCH_RADIUS_CELL = 1
NUM_LATTICE_STEPS = 20
LATTICE_PLOT_PERIOD = 10

VALID_BUILDSTRINGS = ["mono", "dual", "memory_sequence", "random", "explicit"]
VALID_EXOSOME_STRINGS = ["on", "off", "all", "no_exo_field"]  # TODO: implement no_ext_field runtime savings
BUILDSTRING = "dual"
EXOSTRING = "on"
FIELD_REMOVE_RATIO = 0.0
MEANFIELD = False  # global signalling range, all-connected cell-cell neighbour graph (except self-interactions)

AUTOCRINE = False  # cells also signal with themselves (adjacency diagonals are "1")
# TODO implement autocrine in meanfield and non-parallel updating;
#  maybe in lastline get_surroundings_square (method of SpatialCell)
