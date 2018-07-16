import numpy as np

from multicell_constants import VALID_EXOSOME_STRINGS, EXOSTRING
from singlecell.singlecell_constants import BETA, EXT_FIELD_STRENGTH, APP_FIELD_STRENGTH
from singlecell.singlecell_class import Cell
from singlecell.singlecell_functions import state_subsample, state_only_on, state_only_off
from singlecell.singlecell_simsetup import GENE_LABELS, CELLTYPE_LABELS, J


class SpatialCell(Cell):
    def __init__(self, state, label, location, memories_list=CELLTYPE_LABELS, gene_list=GENE_LABELS,
                 state_array=None, steps=None):
        Cell.__init__(self, state, label, memories_list=memories_list, gene_list=gene_list, state_array=state_array,
                      steps=steps)
        self.location = location

    def get_surroundings_square(self, search_radius, gridsize):
        """Specifies the location of the top left corner of the search square
        Args:
            search_radius: half-edge length of the square
        Returns:
            list of locations; length should be (2 * search_radius + 1) ** 2 (- 1 remove self?)
        Notes:
            - periodic BCs apply, so search boxes wrap around at boundaries
            - note that we assert that search_radius be less than half the grid size
            - may have different search radius depending om context (neighbouring bacteria / empty cells)
            - currently DOES NOT remove the original location
        """
        row = self.location[0]
        col = self.location[1]
        surroundings = [[row_to_search % gridsize, col_to_search % gridsize]
                        for row_to_search in xrange(row - search_radius, row + search_radius + 1)
                        for col_to_search in xrange(col - search_radius, col + search_radius + 1)]
        surroundings.remove(self.location)  # TODO test behaviour
        return surroundings

    def get_local_exosome_field(self, lattice, search_radius, gridsize, exosome_string=EXOSTRING, ratio_to_remove=0.0):
        """
        # TODO: try other methods, currently sample from on genes in nearby states
        A - sample from only 'on' genes
        B - sample from whole gene state vector
        """
        neighbours = self.get_surroundings_square(search_radius, gridsize)
        field_state = np.zeros(self.N)
        if exosome_string == "on":
            for loc in neighbours:
                nbr_cell_state = lattice[loc[0]][loc[1]].get_current_state()
                nbr_state_only_on = state_only_on(nbr_cell_state)
                if ratio_to_remove == 0.0:
                    field_state += nbr_state_only_on
                else:
                    nbr_state_only_on = state_subsample(nbr_state_only_on, ratio_to_remove=ratio_to_remove)
                    field_state += nbr_state_only_on
        elif exosome_string == "all":
            for loc in neighbours:
                nbr_cell_state = np.zeros(self.N)
                nbr_cell_state[:] = lattice[loc[0]][loc[1]].get_current_state()[:]
                if ratio_to_remove == 0.0:
                    field_state += nbr_cell_state
                else:
                    nbr_state_subsample = state_subsample(nbr_cell_state, ratio_to_remove=ratio_to_remove)
                    field_state += nbr_state_subsample
        elif exosome_string == "off":
            for loc in neighbours:
                nbr_cell_state = lattice[loc[0]][loc[1]].get_current_state()
                nbr_state_only_off = state_only_off(nbr_cell_state)
                if ratio_to_remove == 0.0:
                    field_state += nbr_state_only_off
                else:
                    nbr_state_only_off = state_subsample(nbr_state_only_off, ratio_to_remove=ratio_to_remove)
                    field_state += nbr_state_only_off
        elif exosome_string == "no_exo_field":
            field_state = np.zeros(self.N)
        else:
            raise ValueError("exosome_string arg invalid, must be one of %s" % VALID_EXOSOME_STRINGS)
        return field_state, neighbours

    def update_with_signal_field(self, lattice, search_radius, gridsize, beta=BETA, intxn_matrix=J, signal_matrix=None, exosome_string=EXOSTRING, ratio_to_remove=0.0,
                                 ext_field_strength=EXT_FIELD_STRENGTH, app_field=None, app_field_strength=APP_FIELD_STRENGTH):
        ext_field, neighbours = self.get_local_exosome_field(lattice, search_radius, gridsize, exosome_string=exosome_string, ratio_to_remove=ratio_to_remove)
        if signal_matrix is not None:
            for loc in neighbours:
                nbr_cell_state = np.zeros(self.N)
                nbr_cell_state[:] = lattice[loc[0]][loc[1]].get_current_state()[:]
                ext_field += np.dot(signal_matrix, nbr_cell_state)

        self.update_state(beta=beta, intxn_matrix=intxn_matrix, ext_field=ext_field, ext_field_strength=ext_field_strength, app_field=app_field, app_field_strength=app_field_strength)