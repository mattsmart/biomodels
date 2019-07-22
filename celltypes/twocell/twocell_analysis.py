import singlecell.init_multiprocessing  # BEFORE numpy
import matplotlib.pyplot as plt
import numpy as np

from singlecell.singlecell_constants import MEMS_MEHTA, MEMS_UNFOLD, BETA, DISTINCT_COLOURS
from singlecell.singlecell_functions import hamiltonian, sorted_energies, label_to_state, get_all_fp, calc_state_dist_to_local_min, partition_basins, reduce_hypercube_dim, state_to_label
from singlecell.singlecell_simsetup import singlecell_simsetup # N, P, XI, CELLTYPE_ID, CELLTYPE_LABELS, GENE_ID
from singlecell.singlecell_visualize import plot_state_prob_map, hypercube_visualize


if __name__ == '__main__':
    # TODO modify code to support two cell state space
    # TODO smaller towcell model? 6 spins 2 memories?
    HOUSEKEEPING = 0
    KAPPA = 2.0

    random_mem = False
    random_W = False
    #simsetup = singlecell_simsetup(unfolding=False, random_mem=random_mem, random_W=random_W, npzpath=MEMS_MEHTA, housekeeping=HOUSEKEEPING)
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, npzpath=MEMS_UNFOLD, housekeeping=HOUSEKEEPING)
    print 'note: N =', simsetup['N']

    DIM = 3
    METHOD = 'diffusion_custom'  # diffusion_custom, spectral_custom
    use_hd = True
    use_proj = True
    plot_X = False
    beta = 1  # 2.0

    exostring = "no_exo_field"  # on/off/all/no_exo_field, note e.g. 'off' means send info about 'off' genes only
    exoprune = 0.0              # amount of exosome field idx to randomly prune from each cell
    gamma = 0.0                 # global EXT_FIELD_STRENGTH tunes exosomes AND sent field
    app_field = None
    if KAPPA > 0 and HOUSEKEEPING > 0:
        app_field = np.zeros(simsetup['N'])
        app_field[-HOUSEKEEPING:] = 1.0
    print app_field

    # get & report energy levels data
    sorted_data, energies = sorted_energies(simsetup, field=app_field, fs=KAPPA)
    fp_annotation, minima, maxima = get_all_fp(simsetup, field=app_field, fs=KAPPA)
    print 'MINIMA labels', minima
    for minimum in minima:
        print minimum, label_to_state(minimum, simsetup['N'])
    print 'MAXIMA labels', maxima
    for maximum in maxima:
        print maximum, label_to_state(maximum, simsetup['N'])
    basins_dict, label_to_fp_label = partition_basins(simsetup, X=None, minima=minima, field=app_field, fs=KAPPA, dynamics='async_fixed')
    for key in basins_dict.keys():
        print key, label_to_state(key, simsetup['N']), len(basins_dict[key]), key in minima
    # reduce dimension
    X_new = reduce_hypercube_dim(simsetup, METHOD, dim=DIM,  use_hd=use_hd, use_proj=use_proj, add_noise=False,
                                 plot_X=plot_X, field=app_field, fs=KAPPA, beta=beta)
    # setup basin colours for visualization
    cdict = {}
    if label_to_fp_label is not None:
        basins_keys = basins_dict.keys()
        assert len(basins_keys) <= 20  # get more colours
        fp_label_to_colour = {a: DISTINCT_COLOURS[idx] for idx, a in enumerate(basins_keys)}
        cdict['basins_dict'] = basins_dict
        cdict['fp_label_to_colour'] = fp_label_to_colour
        cdict['clist'] = [0] * (2 ** simsetup['N'])
        for i in xrange(2 ** simsetup['N']):
            cdict['clist'][i] = fp_label_to_colour[label_to_fp_label[i]]
    # setup basin labels depending on npz
    basin_labels = {}
    for idx in xrange(simsetup['P']):
        state = simsetup['XI'][:, idx]
        antistate = state * -1
        label = state_to_label(state)
        antilabel = state_to_label(antistate)
        basin_labels[label] = r'$\xi^%d$' % idx
        basin_labels[antilabel] = r'$-\xi^%d$' % idx
    i = 1
    for label in minima:
        if label not in basin_labels.keys():
            if label == 0:
                basin_labels[label] = r'$S-$'
            elif label == 511:
                basin_labels[label] = r'$S+$'
            else:
                basin_labels[label] = 'spurious: %d' % i
                print 'unlabelled spurious minima %d: %s' % (i, label_to_state(label, simsetup['N']))
            i += 1
    # conditionally plot housekeeping on subspace
    housekeeping_on_labels = []  # TODO cleanup
    for label in xrange(2**simsetup['N']):
        state = label_to_state(label, simsetup['N'])
        substate = state[-HOUSEKEEPING:]
        if np.all(substate == 1.0):
            housekeeping_on_labels.append(label)
    print len(housekeeping_on_labels)
    # visualize with and without basins colouring
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
                        elevate3D=True, edges=True, all_edges=False, surf=False, colours_dict=None, beta=None)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
                        elevate3D=True, edges=False, all_edges=False, surf=True, colours_dict=None, beta=None)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
                        elevate3D=True, edges=True, all_edges=False, surf=False, colours_dict=None, beta=beta)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
                        elevate3D=True, edges=False, all_edges=False, surf=True, colours_dict=None, beta=beta)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
                        elevate3D=True, edges=False, all_edges=False, surf=False, colours_dict=cdict)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
                        elevate3D=True, edges=True, all_edges=False, surf=False, colours_dict=cdict)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
                        elevate3D=False, edges=False, all_edges=False, surf=False, colours_dict=None)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels,
                        elevate3D=False, edges=True, all_edges=False, surf=False, colours_dict=cdict)

    plot_state_prob_map(simsetup, beta=None)
    plot_state_prob_map(simsetup, beta=5.0)
    plot_state_prob_map(simsetup, beta=None, field=app_field, fs=KAPPA)
    plot_state_prob_map(simsetup, beta=1.0, field=app_field, fs=KAPPA)
