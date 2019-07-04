import numpy as np
import os
import random
import matplotlib.pyplot as plt

from twocell_visualize import simple_vis
from multicell.multicell_class import SpatialCell
from multicell.multicell_constants import VALID_EXOSOME_STRINGS
from singlecell.singlecell_constants import MEMS_MEHTA, APP_FIELD_STRENGTH, BETA
from singlecell.singlecell_data_io import run_subdir_setup, runinfo_append
from singlecell.singlecell_fields import construct_app_field_from_genes
from singlecell.singlecell_simsetup import singlecell_simsetup # N, P, XI, CELLTYPE_ID, CELLTYPE_LABELS, GENE_ID


EXOSOME_STRING = 'no_exo_field'
EXOSOME_PRUNE = 0.0
APP_FIELD_STRENGTH = 1.0

# TODO file IO
# TODO new full timeseries visualizations
# TODO compute sindiv energies, interaction term, display in plot somehow neatly?


def twocell_sim(simsetup, num_steps, beta=BETA, exostring=EXOSOME_STRING, exoprune=EXOSOME_PRUNE, gamma=1.0):

    # check args
    assert type(num_steps) is int
    assert exostring in VALID_EXOSOME_STRINGS
    assert 0.0 <= exoprune < 1.0
    assert 0.0 <= gamma < 10.0

    cell_a_init = simsetup['XI'][:, 0]
    cell_b_init = simsetup['XI'][:, 0]
    lattice = [[SpatialCell(cell_a_init, 'Cell A', [0, 0], simsetup),
                SpatialCell(cell_b_init, 'Cell B', [0, 1], simsetup)]]  # list of list to conform to multicell slattice funtions
    cell_A = lattice[0][0]
    cell_B = lattice[0][1]

    simple_vis(lattice, simsetup, 'Initial condition')

    # local fields initaliztion
    neighbours_A = [[0, 1]]
    neighbours_B = [[0, 0]]

    # app fields initialization
    app_field_step = None  # TODO housekeeping applied field; N vs N+M

    for i in xrange(num_steps):
        # TODO could compare against whole model random update sequence instead of this block version

        # update cell A
        total_field_A, _ = cell_A.get_local_exosome_field(lattice, None, None, exosome_string=exostring,
                                                          ratio_to_remove=exoprune, neighbours=neighbours_A)
        if simsetup['FIELD_SEND'] is not None:
            total_field_A += cell_A.get_local_paracrine_field(lattice, neighbours_A, simsetup)
        cell_A.update_state(simsetup['J'], beta=beta,
                            ext_field=total_field_A,
                            ext_field_strength=gamma,
                            app_field=app_field_step,
                            app_field_strength=APP_FIELD_STRENGTH)
        simple_vis(lattice, simsetup, 'Step %dA' % i)
        # update cell B
        total_field_B, _ = cell_B.get_local_exosome_field(lattice, None, None, exosome_string=exostring,
                                                          ratio_to_remove=exoprune, neighbours=neighbours_B)
        if simsetup['FIELD_SEND'] is not None:
            total_field_B += cell_B.get_local_paracrine_field(lattice, neighbours_B, simsetup)
        cell_B.update_state(simsetup['J'], beta=beta,
                            ext_field=total_field_B,
                            ext_field_strength=gamma,
                            app_field=app_field_step,
                            app_field_strength=APP_FIELD_STRENGTH)
        simple_vis(lattice, simsetup, 'Step %dB' % i)


    return


if __name__ == '__main__':
    random_mem = False
    random_W = False
    simsetup = singlecell_simsetup(unfolding=False, random_mem=random_mem, random_W=random_W, npzpath=MEMS_MEHTA)

    steps = 10  # global NUM_LATTICE_STEPS
    beta = 1.0  # 2.0

    exostring = "no_exo_field"  # on/off/all/no_exo_field, note e.g. 'off' means send info about 'off' genes only
    exoprune = 0.0              # amount of exosome field idx to randomly prune from each cell
    gamma = 0.15   # global EXT_FIELD_STRENGTH tunes exosomes AND sent field

    #app_field = construct_app_field_from_genes(IPSC_EXTENDED_GENES_EFFECTS, simsetup['GENE_ID'], num_steps=steps)        # size N x timesteps or None
    app_field = None
    app_field_strength = 0.0  # 100.0 global APP_FIELD_STRENGTH

    twocell_sim(simsetup, steps, beta=beta, exostring=exostring, exoprune=exoprune, gamma=gamma)
