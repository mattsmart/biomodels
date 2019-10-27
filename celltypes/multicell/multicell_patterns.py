import numpy as np
import os
import random
import matplotlib.pyplot as plt

from multicell_constants import GRIDSIZE, SEARCH_RADIUS_CELL, NUM_LATTICE_STEPS, VALID_BUILDSTRINGS, MEANFIELD, \
    VALID_EXOSOME_STRINGS, EXOSTRING, BUILDSTRING, LATTICE_PLOT_PERIOD, FIELD_REMOVE_RATIO
from multicell_lattice import build_lattice_main, get_cell_locations, prep_lattice_data_dict, write_state_all_cells, \
    write_grid_state_int, write_general_arr, read_general_arr
from multicell_metrics import calc_lattice_energy, calc_compression_ratio, get_state_of_lattice
from multicell_visualize import lattice_uniplotter, reference_overlap_plotter, lattice_projection_composite
from singlecell.singlecell_class import Cell
from singlecell.singlecell_constants import EXT_FIELD_STRENGTH, APP_FIELD_STRENGTH, BETA
from singlecell.singlecell_data_io import run_subdir_setup, runinfo_append
from singlecell.singlecell_fields import construct_app_field_from_genes
from singlecell.singlecell_simsetup import singlecell_simsetup # N, P, XI, CELLTYPE_ID, CELLTYPE_LABELS, GENE_ID


def build_lattice_memories(simsetup, M):
    assert M == 144
    sqrtM = 12
    num_y = sqrtM
    num_x = sqrtM
    assert simsetup['P'] >= 2

    def conv_grid_to_vector(grid):
        def label_to_celltype_vec(label):
            cellytpe_idx = simsetup['CELLTYPE_ID'][label]
            return simsetup['XI'][:, cellytpe_idx]
        lattice_vec = np.zeros(M * simsetup['N'])
        for row in xrange(num_y):
            for col in xrange(num_x):
                posn = sqrtM * row + col
                label = grid[row][col]
                celltype_vec = label_to_celltype_vec(label)
                start_spin = posn * simsetup['N']
                end_spin = (posn+1) * simsetup['N']
                lattice_vec[start_spin:end_spin] = celltype_vec
        return lattice_vec

    mem1 = [['mem_A' for _ in xrange(sqrtM)] for _ in xrange(sqrtM)]  # TODO check elsewhere for bug that was here [[0]*a]*b (list of list)
    mem2 = [['mem_A' for _ in xrange(sqrtM)] for _ in xrange(sqrtM)]
    mem3 = [['mem_A' for _ in xrange(sqrtM)] for _ in xrange(sqrtM)]
    # build mem 1 -- number 1 on 12x12 grid
    for row in xrange(num_y):
        for col in xrange(num_x):
            if row in (1,2) and col in range(2,7):
                mem1[row][col] = 'mem_B'
            elif row in range(3,9) and col in (5,6):
                mem1[row][col] = 'mem_A'
            elif row in (10,11) and col in range(2,10):
                mem1[row][col] = 'mem_B'
            else:
                mem1[row][col] = 'mem_A'
    mem1 = conv_grid_to_vector(mem1)
    # build mem 2 -- number 2 on 12x12 grid
    for row in xrange(num_y):
        for col in xrange(num_x):
            if row in (1,2,5,6,9,10) and col in range(2,10):
                mem2[row][col] = 'mem_B'
            if row in (3,4) and col in (8,9):
                mem2[row][col] = 'mem_B'
            if row in (7,8) and col in (2,3):
                mem2[row][col] = 'mem_B'
    mem2 = conv_grid_to_vector(mem2)
    # build mem 3 -- number 3 on 12x12 grid
    for row in xrange(num_y):
        for col in xrange(num_x):
            if row in (1,2,5,6,9,10) and col in range(2,10):
                mem3[row][col] = 'mem_B'
            if row in (3,4,7,8) and col in (8,9):
                mem3[row][col] = 'mem_B'
    mem3 = conv_grid_to_vector(mem3)
    # slot into array
    lattice_memories = np.zeros((M * simsetup['N'], 3))
    lattice_memories[:, 0] = mem1
    lattice_memories[:, 1] = mem2
    lattice_memories[:, 2] = mem3
    return lattice_memories


def hopfield_on_lattice_memories(simsetup, M, lattice_memories):
    xi = lattice_memories
    corr_matrix = np.dot(xi.T, xi) / float(xi.shape[0])
    print xi[6 * (0):6 * (1), :]
    print 'and'
    print xi[6*(17):6*(18),:]
    corr_inv = np.linalg.inv(corr_matrix)
    intxn_matrix = reduce(np.dot, [xi, corr_inv, xi.T]) / float(xi.shape[0])
    intxn_matrix = intxn_matrix - np.kron(np.eye(M), np.ones((simsetup['N'], simsetup['N'])))
    return intxn_matrix


def sim_lattice_as_cell(num_steps):
    # specify single cell model
    random_mem = False
    random_W = False
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, curated=True, housekeeping=0)
    # build multicell intxn matrix
    gamma = 1.0
    M = 144
    total_spins = M * simsetup['N']
    lattice_memories = build_lattice_memories(simsetup, M)
    lattice_memories_list = ['%d' % idx for idx, elem in enumerate(lattice_memories)]
    lattice_gene_list =['site_%d' % idx for idx in xrange(total_spins)]
    lattice_intxn_matrix = np.kron(np.eye(M), simsetup['J']) + \
                           gamma * (hopfield_on_lattice_memories(simsetup, M, lattice_memories))
    # initialize
    init_cond = np.random.randint(0,2,total_spins)*2 - 1  # TODO alternatives
    lattice_as_cell = Cell(init_cond, 'lattice_as_cell', lattice_memories_list, lattice_gene_list, state_array=None, steps=None)
    # simulate for t steps
    for t in xrange(num_steps):
        print 'step', t
        lattice_as_cell.update_state(lattice_intxn_matrix, beta=BETA, app_field=None,
                                     app_field_strength=APP_FIELD_STRENGTH, async_batch=True)
    # statistics and plots
    # TODO


if __name__ == '__main__':
    sim_lattice_as_cell(10)
    print 'Done'

    # TODO housekeeping field
    """
    app_field = None
    # housekeeping genes block
    KAPPA = 10.0
    if KAPPA > 0:
        # housekeeping auto (via model extension)
        app_field = np.zeros(simsetup['N'])
        if simsetup['K'] > 0:
            app_field[-simsetup['K']:] = 1.0
            print app_field
        else:
            print 'Note gene 0 (on), 1 (on), 2 (on) are HK in A1 memories'
            print 'Note gene 4 (off), 5 (on) are HK in C1 memories'
            app_field[4] = 1.0
            app_field[5] = 1.0
    """
    
    """
    plot_period = 1
    state_int = True
    beta = BETA  # 2.0
    mc_sim(simsetup, gridsize=n, num_steps=steps, buildstring=buildstring, exosome_string=fieldstring,
           field_remove_ratio=fieldprune, ext_field_strength=ext_field_strength, app_field=app_field,
           app_field_strength=KAPPA, beta=beta, plot_period=plot_period, state_int=state_int,
           meanfield=meanfield)
    """

    """
    for beta in [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 4.0, 5.0, 10.0, 100.0]:
        mc_sim(simsetup, gridsize=n, num_steps=steps, buildstring=buildstring, exosome_string=fieldstring,
               field_remove_ratio=fieldprune, ext_field_strength=ext_field_strength, app_field=app_field,
               app_field_strength=app_field_strength, beta=beta, plot_period=plot_period, state_int=state_int, meanfield=meanfield)
    """
