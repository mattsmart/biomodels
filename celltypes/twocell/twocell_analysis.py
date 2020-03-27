import singlecell.init_multiprocessing  # BEFORE numpy
import matplotlib.pyplot as plt
import numpy as np

from singlecell.singlecell_constants import MEMS_MEHTA, MEMS_UNFOLD, BETA, DISTINCT_COLOURS
from singlecell.singlecell_functions import hamiltonian, sorted_energies, label_to_state, get_all_fp, glauber_transition_matrix, partition_basins, reduce_hypercube_dim, state_to_label, spectral_custom
from singlecell.singlecell_simsetup import singlecell_simsetup # N, P, XI, CELLTYPE_ID, CELLTYPE_LABELS, GENE_ID
from singlecell.singlecell_visualize import plot_state_prob_map, hypercube_visualize


def multicell_hamiltonian(simsetup, state, gamma=0.0, app_field=None, kappa=0.0):
    # TODO exosome
    # note state is a N * num_cells vector of gene expression
    N = simsetup['N']
    cell_A = np.copy(state[0:N])
    cell_B = np.copy(state[N:])
    # singlecell terms
    sc_energy_A = hamiltonian(cell_A, simsetup['J'], field=None, fs=0.0)
    sc_energy_B = hamiltonian(cell_B, simsetup['J'], field=None, fs=0.0)
    # (housekeeping) global applied field term
    if app_field is None:
        app_field = np.zeros(N)
    app_field_term = - np.dot(cell_A + cell_B, app_field)
    # interaction terms and extra field
    W = simsetup['FIELD_SEND']
    WdotOne = np.dot(W, np.ones(simsetup['N']))
    WSym2 = (W + W.T)
    intxn_term_1 = - 0.5 * np.dot(cell_A, np.dot(WSym2, cell_B))
    intxn_term_2 = - 0.5 * np.dot(WdotOne, cell_A + cell_B)
    # sum and weight for total energy
    #TODO make app field block form good size
    energy = sc_energy_A + sc_energy_B \
             + gamma * (intxn_term_1 + intxn_term_2) \
             + kappa * app_field_term
    return energy


def build_twocell_J_h(simsetup, gamma, flag_01=True):
    J_singlecell = simsetup['J']
    W_matrix = simsetup['FIELD_SEND']

    # build multicell Jij matrix (2N x 2N)
    numspins = 2 * simsetup['N']
    J_multicell = np.zeros((numspins, numspins))
    block_diag = J_singlecell
    block_offdiag = gamma * W_matrix
    J_multicell[0:simsetup['N'], 0:simsetup['N']] = block_diag
    J_multicell[-simsetup['N']:, -simsetup['N']:] = block_diag
    J_multicell[0:simsetup['N'], -simsetup['N']:] = block_offdiag
    J_multicell[-simsetup['N']:, 0:simsetup['N']] = block_offdiag

    # build multicell applied field vector (2N x 1)
    h_multicell = None
    if flag_01:
        h_multicell = np.zeros(numspins)
        W_dot_one_scaled = np.dot(W_matrix, np.ones(simsetup['N'])) * gamma / 2.0
        h_multicell[0:simsetup['N']] = W_dot_one_scaled
        h_multicell[-simsetup['N']:] = W_dot_one_scaled
    return J_multicell, h_multicell


def refine_applied_field_twocell(N_multicell, h_multicell, housekeeping=0, kappa=0.0, manual_field=None):
    """
    Takes the default multicell app_field which is fixed by the 01-transform of the signalling
    - adds +kappa strength housekeeping field to the housekeeping genes
    - adds manual field as same size pre-defined array
    """
    if N_multicell is None and (housekeeping > 0 or manual_field is not None):
        h_multicell = np.zeros(N_multicell)
    N_singlecell = N_multicell / 2
    if housekeeping > 0 and kappa > 0:
        h_multicell[-HOUSEKEEPING : N_singlecell] += kappa
        h_multicell[2*N_singlecell - HOUSEKEEPING : 2*N_singlecell] += kappa
    if manual_field is not None:
        h_multicell += manual_field
    return h_multicell


def build_multicell_basin_labels(simsetup, N_multicell, minim):
    N_sc = simsetup['N']
    N_multicell = simsetup['N'] * 2
    basin_labels = {}
    for i in xrange(simsetup['P']):
        for j in xrange(simsetup['P']):
            joint_state_pp = np.zeros(N_multicell)
            joint_state_pn = np.zeros(N_multicell)
            # for two cells, have four different joint states depending on the sign (unlike one antistate for one cell)
            joint_state_pp[0:N_sc] = simsetup['XI'][:, i]
            joint_state_pn[0:N_sc] = simsetup['XI'][:, i]
            joint_state_pp[N_sc:] = simsetup['XI'][:, j]
            joint_state_pn[N_sc:] = -1 * simsetup['XI'][:, j]
            joint_state_nn = joint_state_pp * -1
            joint_state_np = joint_state_pn * -1
            # get labels
            label_pp = state_to_label(joint_state_pp)
            label_nn = state_to_label(joint_state_nn)
            label_pn = state_to_label(joint_state_pn)
            label_np = state_to_label(joint_state_np)
            basin_labels[label_pp] = r'$(\xi^%d, \xi^%d)$' % (i, j)
            basin_labels[label_nn] = r'$(-\xi^%d, -\xi^%d)$' % (i, j)
            basin_labels[label_pn] = r'$(\xi^%d, -\xi^%d)$' % (i, j)
            basin_labels[label_np] = r'$(-\xi^%d, \xi^%d)$' % (i, j)
    i = 1
    for label in minima:
        if label not in basin_labels.keys():
            if label == 0:
                basin_labels[label] = r'$S-$'
            elif label == (2**N_multicell - 1):
                basin_labels[label] = r'$S+$'
            else:
                basin_labels[label] = 'spurious: %d' % i
                print 'unlabelled spurious minima %d: %s' % (i, label_to_state(label, N_multicell))
            i += 1
    return basin_labels


def print_fp_info_twocell(simsetup, N_multicell, minima, maxima, energies):
    assert N_multicell == 2 * simsetup['N']
    print 'Minima labels:'
    print minima
    print 'minima label, energy, state vec, overlap vec, proj vec, '
    for minimum in minima:
        minstate = label_to_state(minimum, N_multicell)
        cell_A = minstate[0:simsetup['N']]
        cell_B = minstate[simsetup['N']:]
        print 'state id and energy:', minimum, energies[minimum]
        print'\tA', cell_A, np.dot(simsetup['XI'].T, cell_A) / simsetup['N'], np.dot(simsetup['ETA'], cell_A)
        print '\tB', cell_B, np.dot(simsetup['XI'].T, cell_B) / simsetup['N'], np.dot(simsetup['ETA'], cell_B)
    print '\nMaxima labels:'
    print maxima
    print 'minima label, energy, state vec, overlap vec, proj vec, '
    for maximum in maxima:
        maxstate = label_to_state(maximum, N_multicell)
        cell_A = maxstate[0:simsetup['N']]
        cell_B = maxstate[simsetup['N']:]
        print 'state id and energy:', maximum, energies[maximum]
        print'\tA', cell_A, np.dot(simsetup['XI'].T, cell_A) / simsetup['N'], np.dot(simsetup['ETA'], cell_A)
        print '\tB', cell_B, np.dot(simsetup['XI'].T, cell_B) / simsetup['N'], np.dot(simsetup['ETA'], cell_B)


def build_colour_dict(basins_dict, label_to_fp_label, N_multicell):
    cdict = {}
    if label_to_fp_label is not None:
        basins_keys = basins_dict.keys()
        assert len(basins_keys) <= 20  # get more colours
        fp_label_to_colour = {a: DISTINCT_COLOURS[idx] if idx < len(DISTINCT_COLOURS) else '#000000'
                              for idx, a in enumerate(basins_keys)}  # Note: colours after 20th label are all black
        cdict['basins_dict'] = basins_dict
        cdict['fp_label_to_colour'] = fp_label_to_colour
        cdict['clist'] = [0] * (2 ** N_multicell)
        for i in xrange(2 ** N_multicell):
            cdict['clist'][i] = fp_label_to_colour[label_to_fp_label[i]]
    return cdict


if __name__ == '__main__':
    # model settings
    beta = 6  # 2.0
    GAMMA = 0.5
    NUM_CELLS = 2
    HOUSEKEEPING = 0
    KAPPA = 0.0
    FLAG_01 = False
    assert NUM_CELLS == 2  # try 3 later maybe

    # build singlecell J and the signalling array W
    random_mem = False
    random_W = False
    CURATED = True
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, npzpath=MEMS_UNFOLD,
                                   housekeeping=HOUSEKEEPING, curated=CURATED)
    print 'note: total N = (%d) x %d' % (simsetup['N'], NUM_CELLS)
    N_multicell = simsetup['N'] * NUM_CELLS

    # dynamics and im reduction settings
    DIM_REDUCE = 2
    plot_X = False

    # (housekeeping) applied field preparation
    KAPPA = 0

    # manual applied field on both cells
    manual_field = None

    # prep multicell state space, interaction matrix
    statespace_multicell = np.array([label_to_state(label, N_multicell) for label in xrange(2 ** N_multicell)])
    J_multicell, h_multicell = build_twocell_J_h(simsetup, GAMMA, flag_01=FLAG_01)
    h_multicell = refine_applied_field_twocell(N_multicell, h_multicell, housekeeping=HOUSEKEEPING, kappa=KAPPA,
                                               manual_field=manual_field)

    # get & report energy levels data
    print "\nSorting energy levels, finding extremes..."
    energies, _ = sorted_energies(J_multicell, field=h_multicell, fs=1.0, flag_sort=False)
    fp_annotation, minima, maxima = get_all_fp(J_multicell, field=h_multicell, fs=1.0, statespace=statespace_multicell,
                                               energies=energies)
    print_fp_info_twocell(simsetup, N_multicell, minima, maxima, energies)

    print "\nPartitioning basins..."
    # TODO resolve partitioning and get_all_fp discrepancies
    basins_dict, label_to_fp_label = partition_basins(J_multicell, X=statespace_multicell, minima=minima,
                                                      field=h_multicell, fs=1.0, dynamics='async_fixed')
    print "\nMore minima stats"
    print "key, label_to_state(key, simsetup['N']), len(basins_dict[key]), key in minima, energy"
    for key in basins_dict.keys():
        print key, label_to_state(key, N_multicell), len(basins_dict[key]), key in minima, energies[key]

    # setup basin colours for visualization
    cdict = build_colour_dict(basins_dict, label_to_fp_label, N_multicell)

    # setup basin labels depending on npz
    basin_labels = build_multicell_basin_labels(simsetup, N_multicell, minima)

    # TODO  mulicell revise
    """ 
    # reduce dimension (SC script)
    X_new = reduce_hypercube_dim(simsetup, METHOD, dim=DIM, use_hd=use_hd, use_proj=use_proj, add_noise=False,
                                 plot_X=plot_X, field=app_field, fs=KAPPA, beta=beta)
    """
    # reduce dimension via spectral embedding
    X = glauber_transition_matrix(J_multicell, field=h_multicell, fs=1.0, beta=beta, override=0.0, DTMC=False)
    dim_spectral = 20  # use dim >= number of known minima?
    X_lower = spectral_custom(-X, dim_spectral, norm_each=False, plot_evec=False, skip_small_eval=False)
    from sklearn.decomposition import PCA
    X_new = PCA(n_components=DIM_REDUCE).fit_transform(X_lower)

    # TODO  mulicell revise
    # visualize with and without basins colouring
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels, num_cells=2,
                        elevate3D=True, edges=True, all_edges=False, surf=False, colours_dict=None, beta=None)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels, num_cells=2,
                        elevate3D=True, edges=False, all_edges=False, surf=True, colours_dict=None, beta=None)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels, num_cells=2,
                        elevate3D=True, edges=True, all_edges=False, surf=False, colours_dict=None, beta=beta)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels, num_cells=2,
                        elevate3D=True, edges=False, all_edges=False, surf=True, colours_dict=None, beta=beta)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels, num_cells=2,
                        elevate3D=True, edges=False, all_edges=False, surf=False, colours_dict=cdict)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels, num_cells=2,
                        elevate3D=True, edges=True, all_edges=False, surf=False, colours_dict=cdict)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels, num_cells=2,
                        elevate3D=False, edges=False, all_edges=False, surf=False, colours_dict=None)
    hypercube_visualize(simsetup, X_new, energies, minima=minima, maxima=maxima, basin_labels=basin_labels, num_cells=2,
                        elevate3D=False, edges=True, all_edges=False, surf=False, colours_dict=cdict)

    plot_state_prob_map(J_multicell, beta=None)
    plot_state_prob_map(J_multicell, beta=5.0)
    plot_state_prob_map(J_multicell, beta=None, field=h_multicell, fs=1.0)
    plot_state_prob_map(J_multicell, beta=1.0, field=h_multicell, fs=1.0)



