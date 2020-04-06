import numpy as np
from random import random, shuffle

from singlecell_constants import BETA, EXT_FIELD_STRENGTH, APP_FIELD_STRENGTH, MEMS_UNFOLD
from singlecell_linalg import sorted_eig
from singlecell_simsetup import singlecell_simsetup, unpack_simsetup

"""
Conventions follow from Lang & Mehta 2014, PLOS Comp. Bio
- note the memory matrix is transposed throughout here (dim N x p instead of dim p x N)
"""


def hamming(s1, s2):
    """Calculate the Hamming distance between two bit lists"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def hamiltonian(state_vec, intxn_matrix, field=None, fs=0.0):
    """
    fs is applied_field_strength
    """
    if field is None:
        H = -0.5 * reduce(np.dot, [state_vec.T, intxn_matrix, state_vec])
    else:
        H = -0.5 * reduce(np.dot, [state_vec.T, intxn_matrix, state_vec]) - fs * np.dot(state_vec.T, field)
    return H


def internal_field(state, gene_idx, t, intxn_matrix):
    """
    Original slow summation:
    h_1 = 0
    intxn_list = range(0, gene_idx) + range(gene_idx+1, N)
    for j in intxn_list:
        h_1 += J[gene_idx,j] * state[j,t]  # plus some other field terms... do we care for these?
    """
    internal_field = np.dot(intxn_matrix[gene_idx,:], state[:,t])  # note diagonals assumed to be zero (enforce in J definition)
    return internal_field


def glauber_dynamics_update(state, gene_idx, t, intxn_matrix, unirand, beta=BETA, ext_field=None, app_field=None,
                            ext_field_strength=EXT_FIELD_STRENGTH, app_field_strength=APP_FIELD_STRENGTH):
    """
    unirand: pass a uniform 0,1 random number
        - note previously unirand = random() OR unirand = np.random_intel.random() from intel python distribution
    See page 107-111 Amit for discussion on functional form
    ext_field - N x 1 - field external to the cell in a signalling sense; exosome field in multicell sym
    ext_field_strength  - scaling factor for ext_field
    app_field - N x 1 - unnatural external field (e.g. force TF on for some time period experimentally)
    app_field_strength - scaling factor for appt_field
    """
    total_field = internal_field(state, gene_idx, t, intxn_matrix=intxn_matrix)
    if ext_field is not None:
        total_field += ext_field_strength * ext_field[gene_idx]
    if app_field is not None:
        total_field += app_field_strength * app_field[gene_idx]
    prob_on_after_timestep = 1 / (1 + np.exp(-2*beta*total_field))  # probability that site i will be "up" after the timestep
    if prob_on_after_timestep > unirand:
        state[gene_idx, t] = 1.0
    else:
        state[gene_idx, t] = -1.0
    return state


def state_subsample(state_vec, ratio_to_remove=0.5):
    state_subsample = np.zeros(len(state_vec))
    state_subsample[:] = state_vec[:]
    idx_to_remove = np.random.choice(range(len(state_vec)), int(np.round(ratio_to_remove*len(state_vec))), replace=False)
    for idx in idx_to_remove:
        state_subsample[idx] = 0.0
    return state_subsample


def state_burst_errors(state_vec, ratio_to_flip=0.02):
    state_burst_errors = np.zeros(len(state_vec))
    state_burst_errors[:] = state_vec[:]
    idx_to_flip = np.random.choice(range(len(state_vec)), int(np.round(ratio_to_flip*len(state_vec))), replace=False)
    for idx in idx_to_flip:
        state_burst_errors[idx] = -state_vec[idx]
    return state_burst_errors


def state_only_on(state_vec):
    state_only_on = np.zeros(len(state_vec))
    for idx, val in enumerate(state_vec):
        if val < 0.0:
            state_only_on[idx] = 0.0
        else:
            state_only_on[idx] = val
    return state_only_on


def state_only_off(state_vec):
    state_only_off = np.zeros(len(state_vec))
    for idx, val in enumerate(state_vec):
        if val > 0.0:
            state_only_off[idx] = 0.0
        else:
            state_only_off[idx] = val
    return state_only_off


def state_memory_overlap(state_arr, time, N, xi):
    return np.dot(xi.T, state_arr[:, time]) / N


def state_memory_projection(state_arr, time, a_inv, N, xi):
    return np.dot(a_inv, state_memory_overlap(state_arr, time, N, xi))


def single_memory_projection(state_arr, time, memory_idx, eta):
    """
    Given state_array (N genes x T timesteps) and time t, return projection onto single memory (memory_idx) at t
    - this should be faster than performing the full matrix multiplication (its just a row.T * col dot product)
    - this should be faster if we want many single memories, say less than half of num memories
    """
    return np.dot(eta[memory_idx,:], state_arr[:,time])


def single_memory_projection_timeseries(state_array, memory_idx, eta):
    """
    Given state_array (N genes x T timesteps), return projection (T x 1) onto single memory specified by memory_idx
    """
    num_steps = np.shape(state_array)[1]
    timeseries = np.zeros(num_steps)
    for time_idx in xrange(num_steps):
        timeseries[time_idx] = single_memory_projection(state_array, time_idx, memory_idx, eta)
    return timeseries


def check_memory_energies(xi, celltype_labels, intxn_matrix):
    # in projection method, expect all to have value -N/2, global minimum value (Mehta 2014)
    # TODO: what is expectation in hopfield method?
    for idx, label in enumerate(celltype_labels):
        mem = xi[:,idx]
        h = hamiltonian(mem, intxn_matrix)
        print idx, label, h
    return


def state_to_label(state):
    # Idea: assign integer label (0 to 2^N - 1) to the state
    # state acts like binary representation of integers
    # "0" corresponds to all -1
    # 2^N - 1 corresponds to all +1
    label = 0
    bitlist = (1+np.sign(np.array(state, dtype=int)))/2  # was np.array, now np.sign to handle xi corruption experiments
    for bit in bitlist:
        label = (label << 1) | bit
    return label


def label_to_state(label, N, use_neg=True):
    # n is the integer label of a set of spins
    bitlist = [1 if digit=='1' else 0 for digit in bin(label)[2:]]
    if len(bitlist) < N:
        tmp = bitlist
        bitlist = np.zeros(N, dtype=int)
        bitlist[-len(tmp):] = tmp[:]
    if use_neg:
        state = np.array(bitlist)*2 - 1
    else:
        state = np.array(bitlist)
    return state


def sorted_energies(intxn_matrix, field=None, fs=0.0, flag_sort=True):
    N = intxn_matrix.shape[0]
    num_states = 2 ** N
    energies = np.zeros(num_states)
    for label in xrange(num_states):
        state = label_to_state(label, N, use_neg=True)
        energies[label] = hamiltonian(state, intxn_matrix, field=field, fs=fs)

    sorted_data = None
    if flag_sort:
        energies_ranked = np.argsort(energies)
        sorted_data = {}
        last_rank = 0
        last_energy = 0
        for rank, idx in enumerate(energies_ranked):
            energy = energies[idx]
            if np.abs(energy - last_energy) < 1e-4:
                sorted_data[last_rank]['labels'].append(idx)
                sorted_data[last_rank]['ranks'].append(idx)
            else:
                sorted_data[rank] = {'energy': energy, 'labels': [idx], 'ranks': [rank]}
                last_rank = rank
                last_energy = energy
    return energies, sorted_data


def get_all_fp(intxn_matrix, field=None, fs=0.0, statespace=None, energies=None, inspection=False):
    # TODO 1 - is it possible to partition all 2^N into basins? are many of the points ambiguous where they wont roll into one basin but multiple?
    N = intxn_matrix.shape[0]
    num_states = 2 ** N

    if statespace is None:
        statespace = np.array([label_to_state(label, N) for label in xrange(num_states)])
    if energies is None:
        energies = np.zeros(num_states)
        for label in xrange(num_states):
            energies[label] = hamiltonian(statespace[label,:], intxn_matrix, field=field, fs=fs)

    minima = []
    maxima = []
    fp_annotation = {}
    for label in xrange(num_states):
        is_fp, is_min = check_min_or_max(intxn_matrix, statespace[label,:], energy=energies[label], field=field, fs=fs,
                                         inspection=inspection)
        if is_fp:
            if is_min:
                minima.append(label)
            else:
                maxima.append(label)
            fp_info = [0 for _ in xrange(N)]
            for idx in xrange(N):
                nbr_state = np.copy(statespace[label, :])
                nbr_state[idx] = -1 * nbr_state[idx]
                nbr_label = state_to_label(nbr_state)
                fp_info[idx] = energies[label] <= energies[nbr_label]  # higher or equal energy after flip -> True, else False (nbr is lower)
            fp_annotation[label] = fp_info
    return fp_annotation, minima, maxima


def calc_state_dist_to_local_min(simsetup, minima, X=None, norm=True):
    N = simsetup['N']
    num_states = 2 ** N
    if X is None:
        X = np.array([label_to_state(label, N) for label in xrange(num_states)])
    minima_states = np.array([label_to_state(a, N) for a in minima])
    overlaps = np.dot(X, minima_states.T)
    hamming_dist = 0.5 * (N - overlaps)
    if norm:
        hamming_dist = hamming_dist / N
    return hamming_dist


def check_min_or_max(intxn_matrix, state, energy=None, field=None, fs=0.0, inspection=False):
    # 1) is it a fixed point of the deterministic dynamics?
    N = intxn_matrix.shape[0]

    is_fp = False
    field_term = 0
    if field is not None:
        field_term = field * fs
    total_field = np.dot(intxn_matrix, state) + field_term

    # TODO speedup
    if np.array_equal(np.sign(total_field), np.sign(state)):
        is_fp = True
        if inspection:
            print "\nis_fp verify", state
            print np.sign(total_field), total_field
            print np.sign(state), state

    # 2) is it a min or a max?
    is_min = None
    if is_fp:
        print 'Warning: min max check only looks at first spin (use inspection flag)'
        state_perturb = np.zeros(state.shape)
        state_perturb[:] = state[:]
        state_perturb[0] = -1 * state[0]
        energy_perturb = hamiltonian(state_perturb, intxn_matrix, field, fs)
        if energy is None:
            energy = hamiltonian(state, intxn_matrix, field, fs)
        if energy > energy_perturb:
            is_min = False
        else:
            is_min = True

        if inspection:
            print "check_min_or_max(): state", state
            print 'checking... (TODO remove this testing block)'
            utilvec = np.zeros(N)
            for idx in xrange(N):
                state_perturb = np.zeros(state.shape)
                state_perturb[:] = state[:]
                state_perturb[idx] = -1 * state[idx]  # TODO this is very local perturbation -- just first spin... is it OK?
                energy_perturb = hamiltonian(state_perturb, intxn_matrix, field, fs)
                if np.abs(energy - energy_perturb) < 1e-3:
                    utilvec[idx] = 0
                    ll = 'flip equal'
                elif energy < energy_perturb:
                    utilvec[idx] = 1
                    ll = 'flip higher'
                else:
                    utilvec[idx] = -1
                    ll = 'flip lower'
                print idx, energy, energy_perturb, ll
            print 'state summary: (ismin, ismax)', (utilvec>0).all(), (utilvec<0).all(), utilvec

    return is_fp, is_min


def fp_of_state(intxn_matrix, state_start, app_field=0, dynamics='sync', zero_override=False):
    # TODO cycle support
    """
    Given a state e.g. (1,1,1,1, ... 1) i.e. hypercube vertex, return the corresponding FP of specified dynamics
    """
    assert dynamics in ['sync', 'async_fixed', 'async_batch']

    def safe_field_sign_update_sync(state_current, total_field):
        # tricky parallel site update with check for 0 total field
        # TODO check bugfix for case where an element is zero - try flip sites which have zero field
        # out_template = np.copy(state_current)
        out_template = np.zeros(intxn_matrix.shape[0])
        out_template[:] = state_current[:]
        state_next = np.sign(total_field, where=(total_field != 0), out=-out_template)
        return state_next

    i = 0
    sites = range(intxn_matrix.shape[0])
    state_next = np.copy(state_start)
    state_current = np.zeros(state_start.shape)
    if zero_override:
        print 'Warning fp_of_state() flag "zero override" can lead to bugs'
        # TODO characterize this choice; affects basin characterization confirmed w memories in block form +--, -+-, ---
        # TODO alternative override is to always flip the site if total field is zero
        # When the internal field on a site is zero, it is naturally a coin flip whether to put that spin up or down.
        # We should not have FPs which have 0 total field on one of the spins
        override_sign = 1
        perturb_field = np.ones(intxn_matrix.shape[0]) * 1e-8 * override_sign
        if app_field is None:
            app_field = perturb_field
        else:
            app_field = app_field + perturb_field

    if dynamics == 'sync':
        assert dynamics != 'sync'  # too unpredictable
        # TODO bug if sign of total field is zero
        # TODO cycle support
        # TODO how to handle the flickering/oscillation in sync mode? store extra state, catch 2 cycle, and impute FP?
        CYCLE_THRESHOLD = 2 ** intxn_matrix.shape[0]  # i.e. 2^N
        state_back_two = np.zeros(intxn_matrix.shape[0])  # To catch 2-state flickering
        while not np.array_equal(state_next, state_current):
            i += 1
            state_back_two = state_current
            state_current = state_next
            total_field = np.dot(intxn_matrix, state_next) + app_field
            state_next = safe_field_sign_update_sync(state_current, total_field)

            if np.array_equal(state_back_two, state_next):
                # This is a flicker, so pick the one with lower energy
                energy_next = hamiltonian(state_next, intxn_matrix, field=app_field, fs=1.0)
                energy_current = hamiltonian(state_current, intxn_matrix, field=app_field, fs=1.0)
                print '\nWARNING - sync dynamics fp_of_state() has 2-state flicker, for', state_start
                if state_to_label(state_start) == 3:
                    print 'state_to_label(state_start) == 3:'
                    print energy_next, energy_current
                    print state_to_label(state_next), state_to_label(state_current)

                if energy_next > energy_current:
                    state_next = state_current
                break

            if i > CYCLE_THRESHOLD:
                # This is to catch longer cycles, which have max size of 2^N -- the state space
                state_next = None
                break

    else:
        assert dynamics in ['async_batch', 'async_fixed']
        while not np.array_equal(state_next, state_current):
            if dynamics == 'async_batch':
                shuffle(sites)  # randomize site ordering each update
            state_current = np.copy(state_next)
            for i in sites:
                total_field_on_i = np.dot(intxn_matrix[i, :], state_next) + app_field[i]
                if total_field_on_i == 0:
                    state_next[i] = state_next[i] * -1
                else:
                    state_next[i] = np.sign(total_field_on_i)
                i += 1
    fp = state_next
    return fp


def partition_basins(intxn_matrix, X=None, minima=None, field=None, fs=0.0, dynamics='sync'):
    # TODO cycle attractor support
    assert dynamics in ['sync', 'async_fixed', 'async_batch']

    if minima is None:
        _, minima, _ = get_all_fp(intxn_matrix, field=field, fs=fs)

    N = intxn_matrix.shape[0]
    num_states = 2 ** N

    if field is not None:
        app_field = field * fs
    else:
        app_field = np.zeros(N)

    basins_dict = {label: [] for label in minima}
    label_to_fp_label = {}
    if X is None:
        X = np.array([label_to_state(label, N) for label in xrange(num_states)])

    for label in xrange(num_states):
        state = X[label, :]
        fp = fp_of_state(intxn_matrix, state, app_field=app_field, dynamics=dynamics)
        fp_label = state_to_label(fp)
        if fp_label in minima:
            basins_dict[fp_label].append(label)
        else:
            print "WARNING -- fp_label not in minima;", label, 'went to', fp_label
            print '\t', state, 'went to', fp
            if fp_label in basins_dict.keys():
                basins_dict[fp_label].append(label)
            else:
                basins_dict[fp_label] = [label]
        label_to_fp_label[label] = fp_label
    return basins_dict, label_to_fp_label


def glauber_transition_matrix(intxn_matrix, field=None, fs=0.0, beta=BETA, override=0.0, DTMC=False):
    # TODO why are diagonals so large?  ~ 0.5 as temp -> infty? we see the global minima go to 1 as T->0

    # TODO is it prob per unit time i.e. rate or is it prob in fixed time?
    # TODO what to do for NaN beta? sign fn on total_field
    # TODO it need not be symmetric but spectral is making it symmetric how to fix this
    """
    -Confirmed that in zero T limit the top eigenevectors for unfold C1 all correspond to the known global minima
    directly with a single 1 all else 0 in the 2^N states, and have eval 0.
    -Ongoing issue with C1: xi 1 and xi 2 completely overlap in spectral embedding, 2D or 3D, varying beta -- why?
    """
    N = intxn_matrix.shape[0]
    num_states = 2 ** N
    choice_factor = 1.0 / N
    M = np.zeros((num_states, num_states))
    states = np.array([label_to_state(label, N) for label in xrange(2 ** N)])
    if field is None:
        app_field = np.ones(N) * override
    else:
        app_field = field * fs + override
    for i in xrange(num_states):
        state_end = states[i, :]
        for idx in xrange(N):
            # flip the ith spin
            state_start = np.copy(state_end)
            state_start[idx] = state_start[idx] * -1
            j = state_to_label(state_start)
            # compute glauber_factor
            site_end = state_end[idx]
            total_field_site_start = np.dot(intxn_matrix[idx, :], state_start) + app_field[idx]
            if beta is None:
                if np.sign(total_field_site_start) == site_end:
                    glauber_factor = 1
                else:
                    glauber_factor = 0
            else:
                glauber_factor = 1 / (1 + np.exp(-2 * beta * total_field_site_start * site_end))
            """
            # compute glauber_factor V1 - has expeted structure for B1 C1 but FP overlap for C1...
            total_field = np.dot(intxn_matrix[idx, :], state_end) + app_field[idx]
            if beta is None:
                if np.sign(total_field) == state_end[idx]:
                    glauber_factor = 1
                else:
                    glauber_factor = 0
            else:
                glauber_factor = 1 / (1 + np.exp(-2 * beta * total_field))
            """
            M[i, j] = choice_factor * glauber_factor
    # normalize column sum to 1 if not DTMC i.e. if it is a stoch rate matrix, CTMC
    if DTMC:
        for j in xrange(num_states):
            M[j, j] = 1-np.sum(M[:, j])  # TODO think this normalization is sketchy
            #print j, M[j, j]
    else:
        for j in xrange(num_states):
            M[j, j] = -np.sum(M[:, j])
            #print j, M[j, j]
    return M


def spectral_custom(L, dim, norm_each=False, plot_evec=False, skip_small_eval=False):
    # see https://github.com/hlml-toronto/machinelearning/blob/master/guides/unsupervised/spectral.ipynb
    evals, evecs = sorted_eig(L, take_real=True)

    if plot_evec:
        statevol = evecs.shape[0]
        import matplotlib.pyplot as plt
        for i in xrange(dim):
            plt.plot(range(statevol), evecs[:, i], label='evec %d' % i)
        plt.legend()
        plt.show()
    print evals[0:6]
    print evals[-6:]
    # get first dim evecs, sorted
    if skip_small_eval:
        start_idx = 0
        while evals[start_idx] < 1e-13:
            start_idx += 1
        dim_reduced = evecs[:, start_idx:dim+start_idx]
    else:
        dim_reduced = evecs[:, 0:dim]

    # normalize column sum to 1
    for j in xrange(dim):
        print 'TTT', np.sum(dim_reduced[:, j]), np.sum(np.abs(dim_reduced[:, j]))
        #dim_reduced[:, j] = dim_reduced[:, j] / np.sum(dim_reduced[:, j])
    # normalize the reduced vectors
    # TODO alternative normalize to lie on hypersphere...? Ng 2002
    if norm_each:
        for j in xrange(dim_reduced.shape[0]):
            norm = np.linalg.norm(dim_reduced[j,:])
            if norm > 0:
                dim_reduced[j,:] = dim_reduced[j,:] / norm
    return dim_reduced


def distances_from_master_eqn(X):
    dists = np.zeros(X.shape)
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            x = X[i, j]
            if i == j:
                dists[i, j] = 0
            elif x == 0:
                dists[i, j] = 999
            else:
                dists[i, j] = 1 / x
    return dists


def reduce_hypercube_dim(simsetup, method, dim=2,  use_hd=False, use_proj=False, add_noise=False, plot_X=False,
                         field=None, fs=0.0, beta=BETA, print_XI=True):
    # TODO in addition to hamming dist (i.e. m(s)) should get memory proj a(s)...
    # TODO spectral clustering from MSE with temp?
    # TODO implement own diffusion map since packages wont work (mapalign fail, pydiff no precompute)
    # TODO try umap on its own and in spectral_custom

    N = simsetup['N']
    states = np.array([label_to_state(label, simsetup['N']) for label in xrange(2 ** N)])
    X = states
    if use_hd:
        # Option 1
        """
        fp_annotation, minima, maxima = get_all_fp(simsetup, field=field, fs=fs)
        hd = calc_state_dist_to_local_min(simsetup, minima, X=X)
        """
        # Option 2
        encoded_minima = [state_to_label(simsetup['XI'][:,a]) for a in xrange(simsetup['P'])]
        hd = calc_state_dist_to_local_min(simsetup, encoded_minima, X=states, norm=True)
        X = hd
    if use_proj:
        projdist = np.dot(states, simsetup['ETA'].T)
        if use_hd:
            X = np.concatenate((hd, projdist), axis=1)
        else:
            X = projdist
    if method in ['spectral_auto', 'spectral_custom', 'diffusion']:
        X = glauber_transition_matrix(simsetup['J'], field=field, fs=fs, beta=beta, override=0)
    if plot_X:
        import matplotlib.pyplot as plt
        im = plt.imshow(X, aspect='auto')
        plt.colorbar(im)
        plt.show()

    print 'Performing dimension reduction (%s): (%d x %d) to (%d x %d)' % (method, X.shape[0], X.shape[1], X.shape[0], dim)
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim)
        X_new = pca.fit_transform(X)
    elif method == 'mds':
        from sklearn.manifold import MDS
        # simple call
        statespace = 2 ** N
        dists = np.copy((statespace, statespace), dtype=int)
        for i in xrange(statespace):
            for j in xrange(i):
                d = hamming(X[i, :], X[j, :])
                dists[i, j] = d
        dists = dists + dists.T - np.diag(dists.diagonal())
        X_new = MDS(n_components=dim, max_iter=300, verbose=1, dissimilarity='precomputed').fit_transform(dists)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        perplexity_def = 5.0
        tsne = TSNE(n_components=dim, init='random', random_state=0, perplexity=perplexity_def)
        X_new = tsne.fit_transform(X)
    elif method == 'spectral_auto':
        from sklearn.manifold import SpectralEmbedding
        embedding = SpectralEmbedding(n_components=dim, random_state=0, affinity='precomputed')
        #TODO pass through beta and field, note glauber may be ill-defined at 0 T / beta infty
        X_new = embedding.fit_transform(X.T)
    elif method == 'spectral_custom':
        dim_spectral = 3  # use dim >= number of known minima?
        X_lower = spectral_custom(-X, dim_spectral, norm_each=False, plot_evec=False, skip_pss=True)
        from sklearn.decomposition import PCA
        X_new = PCA(n_components=dim).fit_transform(X_lower)
        """
        from sklearn.manifold import TSNE
        X_new = TSNE(n_components=dim, init='random', random_state=0, perplexity=5.0).fit_transform(X_lower)
        """
    elif method == 'diffusion_custom':
        # TODO cleanup see yale paper
        dim_spectral = 3
        num_steps = 1
        X_DTMC = glauber_transition_matrix(simsetup['J'], field=field, fs=fs, beta=beta, override=0, DTMC=True)
        X_bigstep = np.linalg.matrix_power(X_DTMC, num_steps)
        X_lower = spectral_custom(-X_bigstep, dim_spectral, norm_each=False, plot_evec=False, skip_pss=True)
        from sklearn.decomposition import PCA
        X_new = PCA(n_components=dim).fit_transform(X_lower)
        #X_new = DiffusionMapEmbedding(alpha=0.5, diffusion_time=1, affinity='precomputed',
        #                              n_components=dim).fit_transform(X.copy())
        """
        from pydiffmap import diffusion_map as dm
        neighbor_params = {'affinity': 'precomputed'}
        embedding = dm.DiffusionMap.from_sklearn(n_evecs=2, k=200, epsilon='bgh', alpha=1.0, neighbor_params=neighbor_params)
        # fit to data and return the diffusion map.
        X_new = embedding.fit_transform(X.T)
        """
    else:
        print 'method must be in [pca, mds, tsne, spectral_auto, spectral_custom, diffusion, diffusion_custom]'
    if add_noise:
        # jostles the point in case they are overlapping
        X_new += np.random.normal(0, 0.5, X_new.shape)
    if print_XI:
        for i in xrange(simsetup['P']):
            label = state_to_label(simsetup['XI'][:, i])
            antilabel = state_to_label(simsetup['XI'][:, i] * -1)
            print 'XI %d label %d is %s' % (i, label, X_new[label,:])
            print 'anti-XI %d label %d is %s' % (i, antilabel, X_new[antilabel,:])
    return X_new


if __name__ == '__main__':
    simsetup = singlecell_simsetup(unfolding=True, npzpath=MEMS_UNFOLD)
    M = glauber_transition_matrix(simsetup['J'], field=None, fs=0.0, beta=None, override=0.0)
    print M
    E, V = np.linalg.eig(M)
    eig_ranked = np.argsort(E)[::-1]
    print E[eig_ranked[0:10]]
    top_evec = V[:,eig_ranked[0:8]]
    print top_evec.shape
    import matplotlib.pyplot as plt
    plt.imshow(np.real(top_evec))
    plt.show()
    for i in xrange(8):
        cols = top_evec[:, i]
        cc = np.sort(cols)
        am = np.argmax(cols)
        print cc[0:3], cc[-3:]
        print am
    print
    for i in xrange(3):
        print state_to_label(simsetup['XI'][:, i])
        print state_to_label(simsetup['XI'][:, i]*-1)
