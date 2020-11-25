import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from os import sep, path, makedirs
from scipy.linalg import qr

from data_process import image_data_collapse
from settings import DIR_OUTPUT, DIR_MODELS
from RBM_train import load_rbm_hopfield

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = [r'\usepackage{bm}', r'\usepackage{amsmath}']
print(mpl.rcParams["text.usetex"])
#mpl.rcParams['axes.unicode_minus'] = False  # TEMP BUGFIX


def rebuild_R_from_xi_image(xi_image):
    xi_collapsed = image_data_collapse(xi_image)
    Q, R = qr(xi_collapsed, mode='economic')
    return Q, R


def plot_basis_candidate(xcol, idx, outdir, label=''):
    cmap = 'seismic_r'
    norm = mpl.colors.DivergingNorm(vcenter=0.)

    plt.figure()
    plt.imshow(xcol.reshape((28, 28)), cmap=cmap, norm=norm)

    # turn off labels
    ax = plt.gca()
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    # colorbar
    plt.colorbar()

    # plt.title('Basis example: %d %s' % (idx, label))
    plt.savefig(outdir + sep + 'basis_example_%d%s.jpg' % (idx, label))
    plt.close()


def plot_basis_candidate_fancy(xcol, idx, outdir, label=''):
    # generate masked xnol for discrete cmap
    # ref: https://stackoverflow.com/questions/53360879/create-a-discrete-colorbar-in-matplotlib
    # v <= -1.5       = orange
    # -1.5 < v < -0.5 = light orange
    # -0.5 < v < 0.5  = grey
    # 0.5 < v < 1.5   = light blue
    # v > 1.5         = blue
    cmap = mpl.colors.ListedColormap(["firebrick", "salmon", "lightgrey", "deepskyblue", "mediumblue"])
    norm = mpl.colors.BoundaryNorm(np.arange(-2.5, 3), cmap.N)

    # clip the extreme values
    xcol_clipped = xcol
    xcol_clipped[xcol_clipped > 1.5] = 2
    xcol_clipped[xcol_clipped < -1.5] = -2
    img = xcol_clipped.reshape((28, 28))

    # plot prepped image
    plt.figure()
    ims = plt.imshow(img, cmap=cmap, norm=norm)

    # turn off labels
    ax = plt.gca()
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    # colorbar
    plt.colorbar(ims, ticks=np.linspace(-2, 2, 5))

    # plt.title(r'$Basis example: %d %s$' % (idx, label))
    plt.savefig(outdir + sep + 'basis_example_%d%s.jpg' % (idx, label))
    plt.close()


def plot_error_timeseries(error_timeseries, outdir, label='', ylim=None):
    plt.plot(error_timeseries)
    plt.xlabel('iteration')
    # plt.ylabel(r'$||Wx - tanh(\beta Wx)||^2$')
    plt.title('Error over gradient updates %s' % (label))
    print('error_timeseries min/max', np.min(error_timeseries), np.max(error_timeseries))

    if ylim is None:
        plt.savefig(outdir + sep + 'error_%s.jpg' % (label))
    else:
        plt.ylim(ylim)
        plt.savefig(outdir + sep + 'error_%s_ylim.jpg' % (label))
    plt.close()



def binarize_search_one_column(weights, outdir, num=20, beta=100, init=None):
    # TODO DEPRECATED
    # search for single column x_mu st W x_mu is approximately binary
    # condition for binary: Wx = sgn(Wx)
    # soften the problem as Wx = tanh(beta Wx)
    # perform gradient descent on ||Wx - tanh(beta Wx)||^2

    # speedups and aliases
    N, p = weights.shape
    WTW = np.dot(weights.T, weights)

    def get_err(err_vec):
        err = np.dot(err_vec.T, err_vec)
        return err

    def gradient_search(xcol, column, num_steps=200, eta=2*1e-2, plot_all=True):
        # note eta may need to be prop. to beta; 0.1 worked with beta 200
        # performs gradient descent for single basis vector
        # TODO idea for gradient feedback: add terms as basis formed corresponding to 'dot product with basis elements is small'

        err_timeseries = np.zeros(num_steps + 1)

        # large local output dir for gradient traj
        outdir_local = outdir + os.sep + 'num%d_details' % column
        if not os.path.exists(outdir_local):
            os.makedirs(outdir_local)

        def gradient_iterate(xcol):

            # gather terms
            Wx = np.dot(weights, xcol)
            tanhu = np.tanh(beta * Wx)
            err_vec = Wx - tanhu

            # compute gradient
            """ 
            # OLD (incorrect)
            factor_2 = (1 - beta) * np.eye(N) + beta * np.diag(np.diag(tanhu ** 2))
            factor_3 = tanhu - Wx
            gradient = np.dot(2 * weights.T,
                              np.dot(factor_2, factor_3))
            """
            delta = 1 - tanhu ** 2
            factor_2 = err_vec - beta * err_vec * delta
            gradient = np.dot(2 * weights.T, factor_2)
            xout = xcol - gradient * eta

            return xout, Wx, err_vec

        for idx in range(num_steps):
            xcol, Wx, err_vec = gradient_iterate(xcol)
            err_timeseries[idx] = get_err(err_vec)

            if plot_all:
                plot_basis_candidate_fancy(Wx, column, outdir_local, '(iterate_%s_discrete)' % idx)
                plot_basis_candidate(Wx, column, outdir_local, '(iterate_%s)' % idx)

        # compute last element of error (not done in loop)
        Wx = np.dot(weights, xcol)
        tanhu = np.tanh(beta * Wx)
        err_vec = Wx - tanhu
        err_timeseries[num_steps] = get_err(err_vec)

        return xcol, err_timeseries

    # initial guesses for candidate columns of R matrix
    if init is None:
        X = np.random.rand(p, num)*2 - 1  # draw from U(-1,1)
    else:
        assert init.shape == (p, num)
        X = init

    # perform num random searches for basis vector candidates
    for idx in range(num):
        x0 = X[:, idx]
        xcol, err_timeseries = gradient_search(x0, idx)
        plot_error_timeseries(err_timeseries, outdir, 'traj%s' % idx)
        plot_error_timeseries(err_timeseries, outdir, 'traj%s' % idx, ylim=(-10, 200))
        X[:, idx] = xcol

        Wx_final = np.dot(weights, xcol)
        plot_basis_candidate_fancy(Wx_final, idx, outdir, 'final_fancy')
        plot_basis_candidate(Wx_final, idx, outdir, 'final')

    return X


def binarize_search_as_matrix(weights, outdir, num_steps=200, eta=2*1e-2, beta=100, noise=0.0, init=None, xi_baseline=None):
    # search for (p x p) X such that W*X is approximately binary (N x p matrix)
    # condition for binary: W*X = sgn(W*X)
    # soften the problem as W*X = tanh(beta W*X)
    #    define error E = W*X - tanh(beta W*X)
    # perform gradient descent on ||W*X - tanh(beta W*X)||^2 = tr(E * E^T)

    # speedups and aliases
    N, p = weights.shape
    WTW = np.dot(weights.T, weights)

    def get_err(err_matrix):
        err = np.trace(
            np.dot(err_matrix, err_matrix.T)
        )
        return err

    def build_overlaps(X):
        overlaps = np.zeros((p, p))
        for i in range(p):
            x_i = X[:, i]
            for j in range(p):
                x_j = X[:, j]
                overlaps[i, j] = np.dot(X[:, i],
                                        np.dot(WTW, X[:, j]))
        return overlaps

    def gradient_search(X, num_steps=num_steps, eta=eta, noise=noise, plot_all=True):
        # note eta may need to be prop. to beta; 0.1 worked with beta 200
        # performs gradient descent for single basis vector
        # TODO idea for gradient feedback: add terms as basis formed corresponding to 'dot product with basis elements is small'

        err_timeseries = np.zeros(num_steps + 1)
        #ALPHA = 1e-3  # lagrange mult for encouraging basis vector separation

        # large local output dir for gradient traj
        outdir_local = outdir + sep + 'num_details'
        if not path.exists(outdir_local):
            makedirs(outdir_local)

        def gradient_iterate(X, col_by_col=True):

            # gather terms
            WX = np.dot(weights, X)
            tanhu = np.tanh(beta * WX)
            err_matrix = WX - tanhu

            if col_by_col:
                for col in range(p):
                    delta = 1 - tanhu[:, col] ** 2
                    factor_2 = err_matrix[:, col] - beta * err_matrix[:, col] * delta
                    gradient = np.dot(2 * weights.T, factor_2)

                    ##############################################################################
                    #print('grad:', np.mean(gradient), np.linalg.norm(gradient), np.min(gradient), np.max(gradient))
                    ##############################################################################

                    # encourage separation of the near binary vectors (columns of W*X)
                    # TODO look into this lagrange mult problem further
                    """
                    #overlaps = build_overlaps(X)
                    colsum = 0
                    for c in range(p):
                        if c != col:
                            colsum += X[:, c]  # TODO weight them by their magnitude?
                    alternate_obj = np.dot( WTW, colsum)
                    """

                    # compute overall update (binarization gradient + separation gradient)
                    noise_vec = np.random.normal(loc=0, scale=noise,
                                                 size=p)  # saw print(np.min(gradient * eta), np.max(gradient * eta)) in -1.5, 1.5
                    new_xcol = X[:, col] - gradient * eta + noise_vec  # - ALPHA * alternate_obj

                    # magA = np.linalg.norm(gradient)
                    # print('A', magA, eta * magA)
                    # magB = np.linalg.norm(alternate_obj)
                    # print('B', magB, ALPHA * magB)

                    # update X
                    X[:, col] = new_xcol
            else:
                # compute gradient
                delta = 1 - tanhu ** 2
                factor_2 = err_matrix - beta * err_matrix * delta
                gradient = np.dot(2 * weights.T, factor_2)
                X = X - gradient * eta

            return X, WX, err_matrix

        for idx in range(num_steps):
            X, WX, err_matrix = gradient_iterate(X, col_by_col=True)
            err_timeseries[idx] = get_err(err_matrix)

            if plot_all and idx % 10 == 0:
                for col in range(p):
                    candidate = WX[:, col]
                    plot_basis_candidate_fancy(candidate, col, outdir_local, '(iterate_%s_discrete)' % idx)
                    plot_basis_candidate(candidate, col, outdir_local, '(iterate_%s)' % idx)

        # compute last element of error (not done in loop)
        WX = np.dot(weights, X)
        tanhu = np.tanh(beta * WX)
        err_matrix = WX - tanhu
        err_timeseries[num_steps] = get_err(err_matrix)

        return X, err_timeseries

    # initial guesses for candidate columns of R matrix
    if init is None:
        X = np.random.rand(p, p) * 2 - 1  # draw from U(-1,1)
    else:
        assert init.shape == (p, p)
        X = init

    # perform num random searches for basis vector candidates
    x0 = X
    X_final, err_timeseries = gradient_search(x0)
    plot_error_timeseries(err_timeseries, outdir, 'traj')
    plot_error_timeseries(err_timeseries, outdir, 'traj', ylim=(-10, np.min(err_timeseries) * 2))
    WX_final = np.dot(weights, X_final)
    for idx in range(p):
        candidate = WX_final[:, idx]
        candidate_sign = np.sign(candidate)
        candidate_error = candidate - candidate_sign
        plot_basis_candidate_fancy(candidate, idx, outdir, 'final_fancy')
        plot_basis_candidate(candidate, idx, outdir, 'final')
        plot_basis_candidate_fancy(np.sign(candidate), idx, outdir, 'final_sgn')
        plot_basis_candidate(candidate_error, idx, outdir, 'final_error')
        if xi_baseline is not None:
            xi_orig = xi_baseline[:, idx]
            xi_deviation = candidate - xi_orig
            plot_basis_candidate(xi_deviation, idx, outdir, 'final_dev')
            plot_basis_candidate_fancy(xi_deviation, idx, outdir, 'final_fancy_dev')
            plot_basis_candidate_fancy(candidate_sign - np.sign(xi_orig), idx, outdir, 'final_sgn_dev')
    return X_final, WX_final


def binarize_search_shotgun(weights, outdir, num_steps=200, eta=2*1e-2, beta=100, noise=0.0, init=None, xi_baseline=None):
    # search for (p x p) X such that W*X is approximately binary (N x p matrix)
    # condition for binary: W*X = sgn(W*X)
    # soften the problem as W*X = tanh(beta W*X)
    #    define error E = W*X - tanh(beta W*X)
    # perform gradient descent on ||W*X - tanh(beta W*X)||^2 = tr(E * E^T)

    # speedups and aliases
    N, p = weights.shape
    WTW = np.dot(weights.T, weights)

    def get_err(err_matrix):
        err = np.trace(
            np.dot(err_matrix, err_matrix.T)
        )
        return err

    def gradient_search(X, outdir_trial, num_steps=num_steps, eta=eta, noise=noise, plot_all=False):
        # note eta may need to be prop. to beta; 0.1 worked with beta 200
        # performs gradient descent for single basis vector
        # TODO idea for gradient feedback: add terms as basis formed corresponding to 'dot product with basis elements is small'

        err_timeseries = np.zeros(num_steps + 1)
        #ALPHA = 1e-3  # lagrange mult for encouraging basis vector separation

        # large local output dir for gradient traj
        outdir_local = outdir_trial + sep + 'num_details'
        if not path.exists(outdir_local):
            makedirs(outdir_local)

        def gradient_iterate(X, col_by_col=True):

            # gather terms
            WX = np.dot(weights, X)
            tanhu = np.tanh(beta * WX)
            err_matrix = WX - tanhu

            if col_by_col:
                for col in range(p):

                    delta = 1 - tanhu[:, col] ** 2
                    factor_2 = err_matrix[:, col] - beta * err_matrix[:, col] * delta
                    gradient = np.dot(2 * weights.T, factor_2)

                    ##############################################################################
                    #print('grad:', np.mean(gradient), np.linalg.norm(gradient), np.min(gradient), np.max(gradient))
                    ##############################################################################

                    # compute overall update (binarization gradient + separation gradient)
                    noise_vec = np.random.normal(loc=0, scale=noise,
                                                 size=p)  # saw print(np.min(gradient * eta), np.max(gradient * eta)) in -1.5, 1.5
                    new_xcol = X[:, col] - gradient * eta + noise_vec  # - ALPHA * alternate_obj

                    # update X
                    X[:, col] = new_xcol

            return X, WX, err_matrix

        for idx in range(num_steps):
            X, WX, err_matrix = gradient_iterate(X, col_by_col=True)
            err_timeseries[idx] = get_err(err_matrix)

            if plot_all and idx % 10 == 0:
                for col in range(p):
                    candidate = WX[:, col]
                    plot_basis_candidate_fancy(candidate, col, outdir_local, '(iterate_%s_discrete)' % (idx))
                    plot_basis_candidate(candidate, col, outdir_local, '(iterate_%s)' % (idx))

        # compute last element of error (not done in loop)
        WX = np.dot(weights, X)
        tanhu = np.tanh(beta * WX)
        err_matrix = WX - tanhu
        err_timeseries[num_steps] = get_err(err_matrix)

        return X, err_timeseries

    # initial guesses for candidate columns of R matrix
    if init is None:
        X = np.random.rand(p, p) * 2 - 1  # draw from U(-1,1)
    else:
        assert init.shape == (p, p)
        X = init

    # randomize X
    def randomize_X(X0):

        # add noise to each column of X
        X_jitter = np.zeros(X0.shape)
        for col in range(X0.shape[1]):
            print('COLUMN NOPM', col, np.linalg.norm(X0[:, col]))
            #xcol_noise_term = np.random.normal(0.0, scale=0.1)

            mag = 0.1
            xcol_noise_term = mag * (2 * np.random.rand(X0.shape[1]) - 1.0)

            X_jitter[:, col] += xcol_noise_term

        return X_jitter

    # perform num random searches for basis vector candidates
    ntrials = 10
    for trial in range(ntrials):
        x0 = randomize_X(X)

        outdir_trial = outdir + sep + 'trial%d' % trial
        if not path.exists(outdir_trial):
            makedirs(outdir_trial)

        X_final, err_timeseries = gradient_search(x0, outdir_trial)
        plot_error_timeseries(err_timeseries, outdir_trial, 'traj')
        plot_error_timeseries(err_timeseries, outdir_trial, 'traj', ylim=(-10, np.min(err_timeseries) * 2))
        WX_final = np.dot(weights, X_final)
        for idx in range(p):
            candidate = WX_final[:, idx]
            candidate_sign = np.sign(candidate)
            candidate_error = candidate - candidate_sign
            plot_basis_candidate_fancy(candidate, idx, outdir_trial, 'final_fancy')
            plot_basis_candidate(candidate, idx, outdir_trial, 'final')
            plot_basis_candidate_fancy(np.sign(candidate), idx, outdir_trial, 'final_sgn')
            plot_basis_candidate(candidate_error, idx, outdir_trial, 'final_error')
            if xi_baseline is not None:
                xi_orig = xi_baseline[:, idx]
                xi_deviation = candidate - xi_orig
                plot_basis_candidate(xi_deviation, idx, outdir_trial, 'final_dev')
                plot_basis_candidate_fancy(xi_deviation, idx, outdir_trial, 'final_fancy_dev')
                plot_basis_candidate_fancy(candidate_sign - np.sign(xi_orig), idx, outdir_trial, 'final_sgn_dev')
    return None


if __name__ == '__main__':
    bigruns = DIR_OUTPUT + sep + 'archive' + sep + 'big_runs'

    ##############################################
    # MAIN (load weights)
    ##############################################
    EARLYSTEPS = False
    HIDDEN_UNITS = 10

    if EARLYSTEPS:
        fmod = '_earlysteps'
        run_num = 0
        rundir = 'NOVEMBER_fig4_comparisons_alt_inits_p10_1000batch_earlysteps'
        subdir = 'hopfield_10hidden_0fields_2.00beta_1000batch_3epochs_20cdk_1.00E-04eta_1000ais_10ppEpoch'
        weights_fname = 'weights_10hidden_0fields_20cdk_1000stepsAIS_2.00beta.npz'
        objective_fname = 'objective_10hidden_0fields_20cdk_1000stepsAIS_2.00beta.npz'

    else:
        fmod = ''
        run_num = 0
        rundir = 'NOVEMBER_fig4_comparisons_alt_inits_p10_1000batch'
        subdir = 'hopfield_10hidden_0fields_2.00beta_1000batch_70epochs_20cdk_1.00E-04eta_0ais_1ppEpoch'
        weights_fname = 'weights_10hidden_0fields_20cdk_0stepsAIS_2.00beta.npz'
        objective_fname = 'objective_10hidden_0fields_20cdk_0stepsAIS_2.00beta.npz'

    weights_path = bigruns + sep + 'rbm' + sep + rundir + sep + subdir + sep + 'run%d' % run_num + sep + weights_fname
    objective_path = bigruns + sep + 'rbm' + sep + rundir + sep + subdir + sep + 'run%d' % run_num + sep + objective_fname

    weights_obj = np.load(weights_path)
    weights_timeseries = weights_obj['weights']
    objective_obj = np.load(objective_path)
    epochs = objective_obj['epochs']
    iterations = objective_obj['iterations']
    print('weights_timeseries.shape', weights_timeseries.shape)
    print('epochs', epochs)
    print('iterations', iterations)

    fname = 'hopfield_mnist_%d.npz' % HIDDEN_UNITS
    rbm = load_rbm_hopfield(npzpath=DIR_MODELS + sep + 'saved' + sep + fname)
    Q, R_star = rebuild_R_from_xi_image(rbm.xi_image)
    X_star = R_star
    XI_BASELINE = rbm.xi_image.reshape(28**2, -1)
    print(X_star)

    ##############################################
    # MAIN block #2
    ##############################################

    NOTEBOOK_OUTDIR = DIR_OUTPUT + sep + 'ICLR_nb_reversemap'

    iteration_idx_pick = 50   # this will either be over some number of batches (fraction of epoch) or epoch #
    noise = 0 * 1e-1  # consider making noise dynamically propto gradient?
    eta = 5 * 1e-2    # default 2 * 1e-2
    X_guess = False
    initstr = 'Identity'
    assert initstr in ['NA', 'Random', 'Identity', 'Identity28', 'Triangle', 'TriangleScaled']
    if X_guess:
        assert initstr == 'NA'
    num_steps = 100
    beta = 200
    shotgun = False
    if shotgun:
        fmod += '_shotgun'
        assert X_guess

    alt_names = False   # some weights had to be run separately with different naming convention
    lowdin_approx = False
    ais_val = 1000

    # load misc data to get initial transformation guess (R array if hopfield from QR)
    N = 28 ** 2
    if X_guess:
        X0_guess = X_star
        initstr = 'NA'
    else:
        if initstr == 'Identity':
            X0_guess = np.eye(HIDDEN_UNITS, HIDDEN_UNITS)
        elif initstr == 'Identity28':
            X0_guess = np.eye(HIDDEN_UNITS, HIDDEN_UNITS) * 28
        elif initstr == 'Triangle':
            all_ones = np.ones((HIDDEN_UNITS, HIDDEN_UNITS))
            X0_guess = np.triu(all_ones)
        elif initstr == 'TriangleScaled':
            column_scales = np.array([np.sqrt(N / (mu + 1)) for mu in range(HIDDEN_UNITS)])
            all_ones = np.ones((HIDDEN_UNITS, HIDDEN_UNITS)) * column_scales
            X0_guess = np.triu(all_ones)  # the norm of each column should be sqrt(N) = 28
            print([np.linalg.norm(X0_guess[:,mu]) for mu in range(HIDDEN_UNITS)])
        else:
            assert initstr == 'Random'
            X0_guess = None

    # choose weights to study
    weights = weights_timeseries[:, :, iteration_idx_pick]  # each time-point has shape N x p

    """for idx in range(10):
        weightscol = weights[:, idx]
        plt.imshow(weightscol.reshape(28,28))
        plt.colorbar(); plt.show()"""

    if lowdin_approx:
        print(np.min(weights), np.max(weights))
        print('Taking Lowdin approx of the weights')
        u, s, vh = np.linalg.svd(weights, full_matrices=True)  # False
        print('Original singular values:\n', s)
        print(u.shape, s.shape, vh.shape)
        weights = u[:, 0:HIDDEN_UNITS]

    # analysis
    outdir = NOTEBOOK_OUTDIR + sep + 'hopfield%s_iter%d_star%d_init%s_lowdin%d_num%d_beta%.1f_eta%.1E_noise%.1E' % (fmod, iteration_idx_pick, X_guess, initstr, lowdin_approx, num_steps, beta, eta, noise)
    if not path.exists(outdir):
        makedirs(outdir)

    #  binarize_search(weights, outdir, num=10, beta=2000, init=X0_guess)  # OLD WAY --  search vector by vector
    # NEW WAY - do gradient descent to search for p x p matrix at once
    if shotgun:
        binarize_search_shotgun(weights, outdir, num_steps=num_steps, beta=beta, init=X0_guess, noise=noise, eta=eta,
                                xi_baseline=XI_BASELINE)
    else:
        X_final, WX_final = binarize_search_as_matrix(weights, outdir, num_steps=num_steps, beta=beta, init=X0_guess, noise=noise, eta=eta,
                                                      xi_baseline=XI_BASELINE)
        npzname = 'binarized_rbm_hopfield%s_iter%d_star%d_lowdin%d_num%d_beta%.1f_eta%.1E_noise%.1E.npz' % \
                  (fmod, iteration_idx_pick, X_guess, lowdin_approx, num_steps, beta, eta, noise)
        np.savez(outdir + sep + npzname,
                 W=weights,
                 X_final=X_final,
                 WX_final=WX_final)
    print('outdir:\n', outdir)
