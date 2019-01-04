from matplotlib import pyplot as plt
import numpy as np
import os
from random import random

from singlecell_constants import BETA, EXT_FIELD_STRENGTH, APP_FIELD_STRENGTH, MEMS_MEHTA, MEMS_SCMCA, FIELD_PROTOCOL, MEMORIESDIR
from singlecell_functions import hamiltonian
from singlecell_simsetup import singlecell_simsetup, unpack_simsetup
from singlecell_visualize import plot_as_bar


EXPT_FIELDS = {
    # mir 21 field note:
    #   level 1 is main ref
    #   level 2 adds wiki mir21
    #   level 4 adds targetscan hits
    'miR_21': {
        '2014mehta': {
            'level_1': ['Klf5'],
            'level_2': ['Klf5', 'Trp63', 'Mef2c'],
            'level_3': ['Klf5', 'Trp63', 'Mef2c', 'Smarcd1', 'Crebl2', 'Thrb', 'Nfat5', 'Gata2', 'Nkx6-1', 'Terf2',
                        'Zkscan5', 'Glis2', 'Egr3', 'Foxp2', 'Smad7', 'Tbx2', 'Cbx4', 'Myt1l', 'Satb1', 'Yap1', 'Foxp1',
                        'Foxg1', 'Pcbd1', 'Bahd1', 'Bcl11b', 'Pitx2', 'Sox7', 'Sox5', 'Alx1', 'Npas3', 'Adnp', 'Klf6',
                        'Sox2', 'Klf3', 'Msx1', 'Plag1', 'Osr1', 'Mycl1', 'Nfib', 'Nfia', 'Bnc2']},
        '2018scMCA': {
            'level_1': ['Klf5', 'Pten'],
            'level_2': ['Klf5', 'Pten', 'Anp32a', 'Hnrnpk', 'Mef2c', 'Pdcd4', 'Smarca4', 'Trp63'],
            'level_3': ['Klf5', 'Pten', 'Anp32a', 'Hnrnpk', 'Mef2c', 'Pdcd4', 'Smarca4', 'Trp63', 'Adnp', 'Ago2', 'Alx1',
                        'Asf1a', 'Bcl11b', 'Bnc2', 'Cbx4', 'Chd7', 'Cnot6', 'Crebl2', 'Crebrf', 'Csrnp3', 'Egr3', 'Elf2',
                        'Foxg1', 'Foxp1', 'Foxp2', 'Gata2', 'Gatad2b', 'Glis2', 'Hipk3', 'Hnrnpu', 'Kdm7a', 'Klf3',
                        'Klf6', 'Lcor', 'Msx1', 'Mycl', 'Myt1l', 'Nfat5', 'Nfia', 'Nfib', 'Nipbl', 'Nkx6-1', 'Notch2',
                        'Npas3', 'Osr1', 'Pbrm1', 'Pcbd1', 'Pdcd4', 'Peli1', 'Pik3r1', 'Pitx2', 'Plag1', 'Pspc1', 'Pura',
                        'Purb', 'Purg', 'Rbpj', 'Rnf111', 'Satb1', 'Ski', 'Smad7', 'Smarcd1', 'Sox2', 'Sox2ot', 'Sox5',
                        'Sox7', 'Stat3', 'Suz12', 'Tbx2', 'Terf2', 'Thrb', 'Tnks', 'Trim33', 'Wwp1', 'Yap1', 'Zfp36l2',
                        'Zkscan5', 'Zfp367']}
    },
    # yamanaka field notes: Pou5f1 is alias Oct4, these are OSKM + Nanog
    'yamanaka': {
        '2014mehta': {
            'level_1': ['Sox2', 'Pou5f1', 'Klf4', 'Myc'],
            'level_2': ['Sox2', 'Pou5f1', 'Klf4', 'Myc', 'Nanog']},
        '2018scMCA': {
            'level_1': ['Sox2', 'Pou5f1', 'Klf4', 'Myc'],
            'level_2': ['Sox2', 'Pou5f1', 'Klf4', 'Myc', 'Nanog']},
    },
    # empty field list for None protocol
    None: {}
}


def construct_app_field_from_genes(gene_name_effect, gene_id, num_steps=0):
    """
    Args:
    - gene_name_effect: dict of gene_name: +-1 (on or off)
    - gene_id: map of gene name to idx for the input memories file
    - num_steps: optional numsteps (return 2d array if nonzero)
    Return:
    - applied field array of size N x 1 or N x num_steps
    """
    print "Constructing applied field:"
    N = len(gene_id.keys())
    #app_field = np.zeros((N, num_steps))  $ TODO implement time based
    app_field = np.zeros(N)
    for label, effect in gene_name_effect.iteritems():
        #app_field[gene_id[label], :] += effect
        if label in gene_id.keys():
            print label, gene_id[label], 'effect:', effect
            app_field[gene_id[label]] += effect
        else:
            print "Field construction warning: label %s not in gene_id.keys()" % label
    return app_field


def field_setup(simsetup, protocol=FIELD_PROTOCOL, level=None):
    """
    Construct applied field vector (either fixed or on varying under a field protocol) to bias the dynamics
    Notes on named fields
    - Yamanaka factor (OSKM) names in mehta datafile: Sox2, Pou5f1 (oct4), Klf4, Myc, also nanog
    """
    # TODO must optimize: naive implement brings i7-920 row: 16x200 from 56sec (None field) to 140sec (not parallel)
    # TODO support time varying cleanly
    # TODO speedup: initialize at the same time as simsetup
    # TODO speedup: pre-multiply the fields so it need not to be scaled each glauber step (see singlecell_functions.py)
    # TODO there are two non J_ij fields an isolated single cell experiences: TF explicit mod and type biasing via proj
    # TODO     need to include the type biasing one too
    assert protocol in ["yamanaka", "miR_21", None]
    field_dict = {'protocol': protocol,
                  'time_varying': False,
                  'app_field': None,
                  'app_field_strength': 1e5}  # TODO calibrate this to be very large compared to J*s scale
    gene_id = simsetup['GENE_ID']

    # preamble
    if simsetup['memories_path'] == MEMS_MEHTA:
        npz_label = '2014mehta'
    elif simsetup['memories_path'] == MEMS_SCMCA:
        npz_label = '2018scMCA'
    else:
        print "Note npz mems not supported:", simsetup['memories_path']
        npz_label = None
    if level is None:
        print "Warning: Arg 'level' is None -- setting field level to 'level_1'"
        level = 'level_1'

    if protocol == "yamanaka":
        print "Note: field_setup using", protocol, npz_label, level
        field_genes = EXPT_FIELDS[protocol][npz_label][level]
        field_genes_effects = {label: 1.0 for label in field_genes}  # this ensure all should be OFF
        app_field_start = construct_app_field_from_genes(field_genes_effects, gene_id, num_steps=0)
        field_dict['app_field'] = app_field_start
    elif protocol == 'miR_21':
        """
        - 2018 Nature comm macrophage -> fibroblast paper lists KLF-5 and PTEN as primary targets of miR-21
        - 2014 mehta dataset does not contain PTEN, but 2018 scMCA does
        """
        print "Note: field_setup using", protocol, npz_label, level
        field_genes = EXPT_FIELDS[protocol][npz_label][level]
        field_genes_effects = {label: -1.0 for label in field_genes}  # this ensure all should be OFF
        app_field_start = construct_app_field_from_genes(field_genes_effects, gene_id, num_steps=0)
        field_dict['app_field'] = app_field_start
    else:
        assert protocol is None
    return field_dict


if __name__ == '__main__':
    plot_field_impact = True

    if plot_field_impact:
        npz_mehta = MEMORIESDIR + os.sep + '2014_mehta_mems_genes_types_boolean_compressed_pruned_A.npz'
        npz_scmca = MEMORIESDIR + os.sep + '2018_scmca_mems_genes_types_boolean_compressed_pruned_A_TFonly.npz'
        npz_type = '2018scMCA'
        assert npz_type in ['2014mehta', '2018scMCA']
        simsetup = singlecell_simsetup(npzpath=npz_scmca)
        xi_orig = simsetup['XI']
        for field_type in EXPT_FIELDS.keys():
            if field_type is None:
                continue
            field_levels_dict = EXPT_FIELDS[field_type][npz_type]
            for field_level in field_levels_dict.keys():
                plot_subtitle = "Field effect of %s, %s on %s" % (field_type, field_level, npz_type)
                print plot_subtitle
                field_dict = field_setup(simsetup, protocol=field_type, level=field_level)
                app_field_vector = field_dict['app_field']
                xi_under_field = np.zeros(xi_orig.shape)
                print app_field_vector.shape
                for idx in xrange(app_field_vector.shape[0]):
                    if app_field_vector[idx] == 0:
                        xi_under_field[idx, :] = xi_orig[idx, :]
                    else:
                        xi_under_field[idx, :] = app_field_vector[idx]
                energies = np.zeros(xi_orig.shape[1])
                for col in xrange(xi_orig.shape[1]):
                    energies[col] = -hamiltonian(xi_under_field[:, col], simsetup['J'])
                plot_as_bar(energies, simsetup['CELLTYPE_LABELS'])
                plt.title('%s minima depth' % plot_subtitle)
                plt.show()
