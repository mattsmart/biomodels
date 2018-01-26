GRIDSIZE = 4
SEARCH_RADIUS_CELL = 1
assert SEARCH_RADIUS_CELL < GRIDSIZE / 2
NUM_LATTICE_STEPS = 20
LATTICE_PLOT_PERIOD = 10
FIELD_REMOVE_RATIO = 0.0

VALID_BUILDSTRINGS = ["mono", "dual", "memory_sequence"]
VALID_FIELDSTRINGS = ["on", "off", "all", "no_ext_field"]  # TODO: implement no_ext_field runtime savings
BUILDSTRING = "dual"
FIELDSTRING = "on"
