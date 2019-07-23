import singlecell.init_multiprocessing  # BEFORE numpy
import matplotlib.pyplot as plt
import numpy as np

from singlecell.singlecell_constants import MEMS_MEHTA, MEMS_UNFOLD, BETA, DISTINCT_COLOURS
from singlecell.singlecell_functions import hamiltonian, sorted_energies, label_to_state, get_all_fp, glauber_transition_matrix, partition_basins, reduce_hypercube_dim, state_to_label, spectral_custom
from singlecell.singlecell_simsetup import singlecell_simsetup # N, P, XI, CELLTYPE_ID, CELLTYPE_LABELS, GENE_ID
from singlecell.singlecell_visualize import plot_state_prob_map, hypercube_visualize


def multicell_glauber_transition_matrix(simsetup, num_cells, gamma=0, app_field=None, kappa=0, beta=BETA, override=0, DTMC=False):
    """
    For documentation see "glauber_transition_matrix()"
    Note the dynamics here are more conventional than that used in simulating tissue trajectories,
        which use a blocked form where a cell is updated and then fields are recomputed
    Override is small eps param used to break ties for deterministic dynamics (i.e. h=0)
    """
    if num_cells == 1:
        X = glauber_transition_matrix(simsetup, field=app_field, fs=kappa, beta=beta, override=override, DTMC=DTMC)
    else:
        assert num_cells == 2  # todo extend support to multicell
        N = simsetup['N']
        N_multicell = N * num_cells
        num_states = 2 ** N_multicell
        # TODO how to properly crawl the state space? is standard way OK
        states = np.array([label_to_state(label, N_multicell) for label in xrange(2 ** N_multicell)])
        # note we compute cell-cell field block-wise, so first N spins belong to cell 0 etc

        # TODO OVERALL
        # TODO need to do internal and external for each cell before/after spin flip
        # TODO should W be symmetrized or not
        # maybe smarter way to do this H_A - H_B (ie. like the 2 s_k h_k of one cell glauber)
        # Q: what is the s_k h_k here? s_k is clear....

        choice_factor = 1.0 / N_multicell
        X = np.zeros((num_states, num_states))

        # prep global all-cell applied field + override
        # Note: kappa is meant to scale the 'housekeeping field' to delete anti-minima
        if app_field is None:
            app_field_fixed = np.ones(N) * override
        else:
            app_field_fixed = app_field * kappa + override

        # prep cell-cell sent field
        signal_fixed_implicit = np.dot(simsetup['FIELD_SEND'], np.ones(N))
        sent_field_fixed = signal_fixed_implicit

        print 'Building 2 ** %d glauber transition matrix (multicell)' % (N_multicell)
        for i in xrange(num_states):
            state_end = states[i, :]
            for idx in xrange(N_multicell):
                gene_idx = idx % N
                cell_idx = idx // N
                # flip the ith spin
                state_start = np.copy(state_end)
                state_start[idx] = state_start[idx] * -1
                site_end = state_end[idx]
                j = state_to_label(state_start)
                # compute field sent from other cells as N-vector
                other_cell_idx = (cell_idx + 1) % 2
                state_other_cell = np.copy(state_start[other_cell_idx*N : (1 + other_cell_idx)*N])
                sent_field_local = np.dot(simsetup['FIELD_SEND'][gene_idx, :], state_other_cell)
                # compute glauber_factor
                internal_field_on_cell = np.dot(simsetup['J'][gene_idx, :], state_start[cell_idx * N : (1 + cell_idx)*N])
                total_field_site_start = internal_field_on_cell \
                                         + 0.5 * gamma * (sent_field_local + sent_field_fixed[gene_idx]) \
                                         + app_field_fixed[gene_idx]
                if beta is None:
                    if np.sign(total_field_site_start) == site_end:
                        glauber_factor = 1
                    else:
                        glauber_factor = 0
                else:
                    glauber_factor = 1 / (1 + np.exp(-2 * beta * total_field_site_start * site_end))
                X[i, j] = choice_factor * glauber_factor
        # normalize column sum to 1 if not DTMC i.e. if it is a stoch rate matrix, CTMC
        if DTMC:
            for j in xrange(num_states):
                X[j, j] = 1-np.sum(X[:, j])  # TODO think this normalization is sketchy
                #print j, M[j, j]
        else:
            for j in xrange(num_states):
                X[j, j] = -np.sum(X[:, j])
                #print j, M[j, j]
        return X


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


def multicell_energies(simsetup, num_cells, gamma=0.0, app_field=None, kappa=0.0):
    assert num_cells == 2  # todo extend support to multicell
    N = simsetup['N']
    N_multicell = N * num_cells
    num_states = 2 ** N_multicell
    states = np.array([label_to_state(label, N_multicell) for label in xrange(2 ** N_multicell)])
    energies = np.zeros(num_states)
    for i in xrange(num_states):
        energies[i] = multicell_hamiltonian(simsetup, states[i,:], gamma=0.0, app_field=None, kappa=0.0)
    return energies


if __name__ == '__main__':
    # model settings
    beta = 10  # 2.0
    GAMMA = 100
    NUM_CELLS = 2
    HOUSEKEEPING = 1
    KAPPA = 2.0
    assert NUM_CELLS == 2  # try for 3 later maybe

    # simsetup settings
    random_mem = False
    random_W = False
    #simsetup = singlecell_simsetup(unfolding=False, random_mem=random_mem, random_W=random_W, npzpath=MEMS_MEHTA, housekeeping=HOUSEKEEPING)
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, npzpath=MEMS_UNFOLD, housekeeping=HOUSEKEEPING)
    print 'note: total N = (%d) x %d' % (simsetup['N'], NUM_CELLS)
    N_multicell = simsetup['N'] * NUM_CELLS

    # dynamics and im reduction settings
    DIM_REDUCE = 2
    plot_X = False

    # (housekeeping) applied field preparation
    exostring = "no_exo_field"  # on/off/all/no_exo_field, note e.g. 'off' means send info about 'off' genes only
    exoprune = 0.0              # amount of exosome field idx to randomly prune from each cell
    app_field = None
    if KAPPA > 0 and HOUSEKEEPING > 0:
        app_field = np.zeros(simsetup['N'])
        app_field[-HOUSEKEEPING:] = 1.0
    print app_field

    # prep multicell state space
    statespace_multicell = np.array([label_to_state(label, N_multicell) for label in xrange(2 ** N_multicell)])
    # define master equation (multicell version)
    # TODO multicell -- missing exosome field + check for bugs in implementation
    X = multicell_glauber_transition_matrix(simsetup, NUM_CELLS, gamma=GAMMA, app_field=app_field, kappa=KAPPA,
                                            beta=beta, override=0, DTMC=False)
    # reduce dimension via spectral embedding
    dim_spectral = 20  # use dim >= number of known minima?
    X_lower = spectral_custom(-X, dim_spectral, norm_each=False, plot_evec=False, skip_pss=True)
    from sklearn.decomposition import PCA
    X_new = PCA(n_components=DIM_REDUCE).fit_transform(X_lower)

    # get & report energy levels data
    # TODO exosomes
    energies = multicell_energies(simsetup, NUM_CELLS, gamma=GAMMA, app_field=app_field, kappa=KAPPA)

    # TODO multicell version
    fp_annotation, minima, maxima = get_all_fp(simsetup, field=app_field, fs=KAPPA)
    print 'MINIMA labels', minima
    for minimum in minima:
        print minimum, label_to_state(minimum, simsetup['N'])
    print 'MAXIMA labels', maxima
    for maximum in maxima:
        print maximum, label_to_state(maximum, simsetup['N'])

    # TODO multicell version
    basins_dict, label_to_fp_label = partition_basins(simsetup, X=None, minima=minima, field=app_field, fs=KAPPA, dynamics='async_fixed')
    for key in basins_dict.keys():
        print key, label_to_state(key, simsetup['N']), len(basins_dict[key]), key in minima

    # setup basin colours for visualization
    cdict = {}
    if label_to_fp_label is not None:
        basins_keys = basins_dict.keys()
        assert len(basins_keys) <= 20  # get more colours
        fp_label_to_colour = {a: DISTINCT_COLOURS[idx] for idx, a in enumerate(basins_keys)}
        cdict['basins_dict'] = basins_dict
        cdict['fp_label_to_colour'] = fp_label_to_colour
        cdict['clist'] = [0] * (2 ** N_multicell)
        # TODO multicell
        """
        for i in xrange(2 ** N_multicell):
            cdict['clist'][i] = fp_label_to_colour[label_to_fp_label[i]]
        """
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
                print 'unlabelled spurious minima %d: %s' % (i, label_to_state(label, N_multicell))
            i += 1

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
