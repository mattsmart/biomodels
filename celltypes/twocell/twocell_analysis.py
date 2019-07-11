import singlecell.init_multiprocessing  # BEFORE numpy
import numpy as np

from singlecell.singlecell_constants import MEMS_MEHTA, MEMS_UNFOLD, BETA
from singlecell.singlecell_functions import single_memory_projection_timeseries, hamiltonian, sorted_energies, label_to_state
from singlecell.singlecell_simsetup import singlecell_simsetup # N, P, XI, CELLTYPE_ID, CELLTYPE_LABELS, GENE_ID
from singlecell.singlecell_visualize import plot_state_prob_map, hypercube_visualize


if __name__ == '__main__':
    HOUSEKEEPING = 0
    KAPPA = 100

    random_mem = False
    random_W = False
    #simsetup = singlecell_simsetup(unfolding=False, random_mem=random_mem, random_W=random_W, npzpath=MEMS_MEHTA)
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, npzpath=MEMS_UNFOLD,
                                   housekeeping=HOUSEKEEPING)
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, npzpath=MEMS_UNFOLD,
                                   housekeeping=HOUSEKEEPING)
    print 'note: N =', simsetup['N']

    beta = 2.0  # 2.0

    exostring = "no_exo_field"  # on/off/all/no_exo_field, note e.g. 'off' means send info about 'off' genes only
    exoprune = 0.0              # amount of exosome field idx to randomly prune from each cell
    gamma = 0.0                 # global EXT_FIELD_STRENGTH tunes exosomes AND sent field
    app_field = None
    if KAPPA > 0 and HOUSEKEEPING > 0:
        app_field = np.zeros(simsetup['N'])
        app_field[-HOUSEKEEPING:] = 1.0
    print app_field

    # additional visualizations
    # TODO singlecell simsetup vis of state energies
    sorted_data, energies = sorted_energies(simsetup, field=None, fs=0.0)
    print sorted_data.keys()
    print sorted_data[0]
    for elem in sorted_data[0]['labels']:
        state = label_to_state(elem, simsetup['N'])
        print state, hamiltonian(state, simsetup['J']), np.dot(simsetup['ETA'], state)

    hypercube_visualize(simsetup, 'mds', energies=energies, elevate3D=True, edges=True, all_edges=False)

    """
    import matplotlib.pyplot as plt
    plt.imshow(simsetup['J'])
    plt.show()
    print simsetup['J']
    plt.imshow(simsetup['A'])
    plt.show()
    print simsetup['A']
    plt.imshow(simsetup['ETA'])
    plt.show()
    print simsetup['ETA']
    """

    plot_state_prob_map(simsetup, beta=None)
    plot_state_prob_map(simsetup, beta=5.0)
    plot_state_prob_map(simsetup, beta=None, field=app_field, fs=KAPPA)
    plot_state_prob_map(simsetup, beta=1.0, field=app_field, fs=KAPPA)
