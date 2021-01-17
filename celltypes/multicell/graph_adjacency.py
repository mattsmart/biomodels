import numpy as np


def lattice_square_loc_to_int(loc, sidelength):
    # maps a two-tuple, for the location of a cell on square grid, to a unique integer
    # sidelength is sqrt(num_cells), the edge length of the lattice
    x, y = loc[0], loc[1]
    return x * sidelength + y


def lattice_square_int_to_loc(node_idx, sidelength):
    # maps node_idx, the unique int rep of a cell location on the grid, to corresponding two-tuple
    # sidelength is sqrt(num_cells), the edge length of the lattice
    y = node_idx % sidelength              # remainder from the division mod n
    x = int((node_idx - y) / sidelength)   # solve for x
    return x, y


def adjacency_lattice_square(sidelength, num_cells, search_radius):
    assert num_cells == sidelength ** 2
    adjacency_arr_uptri = np.zeros((num_cells, num_cells))
    # build only upper diagonal part of A
    for a in range(num_cells):
        grid_loc_a = lattice_square_int_to_loc(a, sidelength)  # map cell id to grid loc (i, j)
        arow, acol = grid_loc_a[0], grid_loc_a[1]
        arow_low = arow - search_radius
        arow_high = arow + search_radius
        acol_low = acol - search_radius
        acol_high = acol + search_radius
        for b in range(a + 1, num_cells):
            grid_loc_b = lattice_square_int_to_loc(b, sidelength)  # map cell id to grid loc (i, j)
            # is the cell a neighbor?
            if (arow_low <= grid_loc_b[0] <= arow_high) and \
                    (acol_low <= grid_loc_b[1] <= acol_high):
                adjacency_arr_uptri[a, b] = 1
    adjacency_arr_lowtri = adjacency_arr_uptri.T
    adjacency_arr = adjacency_arr_lowtri + adjacency_arr_uptri
    return adjacency_arr


def adjacency_general(num_cells):
    # TODO implement
    return None
