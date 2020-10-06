import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import qr

from data_process import image_data_collapse
from settings import DIR_OUTPUT, DIR_MODELS


def rebuild_R_from_xi_image(xi_image):
    xi_collapsed = image_data_collapse(xi_image)
    Q, R = qr(xi_collapsed, mode='economic')
    return Q, R


def plot_basis_candidate(xcol, idx, outdir, label=''):
    cmap='seismic_r'
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

    plt.title('Basis example: %d %s' % (idx, label))
    plt.savefig(outdir + os.sep + 'basis_example_%d%s.jpg' % (idx, label))
    plt.close()


def plot_basis_candidate_fancy(xcol, idx, outdir, label=''):

    # generate masked xnol for discrete cmap
    # ref: https://stackoverflow.com/questions/53360879/create-a-discrete-colorbar-in-matplotlib
    # v <= -1.5       = orange
    # -1.5 < v < -0.5 = light orange
    # -0.5 < v < 0.5  = grey
    # 0.5 < v < 1.5   = light blue
    # v > 1.5         = blue
    cmap = mpl.colors.ListedColormap(["firebrick", "salmon", "lightgrey", "deepskyblue",  "mediumblue"])
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

    plt.title('Basis example: %d %s' % (idx, label))
    plt.savefig(outdir + os.sep + 'basis_example_%d%s.jpg' % (idx, label))
    plt.close()


def plot_error_timeseries(error_timeseries, outdir, label='', ylim=None):
    plt.plot(error_timeseries)
    plt.xlabel('iteration')
    plt.ylabel(r'$||Wx - tanh(\beta Wx)||^2$')
    plt.title('Error over gradient updates %s' % (label))

    if ylim is None:
        plt.savefig(outdir + os.sep + 'error_%s.jpg' % (label))
    else:
        plt.ylim(ylim)
        plt.savefig(outdir + os.sep + 'error_%s_ylim.jpg' % (label))
    plt.close()


def binarize_search(weights, outdir, num=20, beta=100, init=None):
    # search for x_mu st W x_mu is approximately binary
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


def binarize_search_as_matrix(weights, outdir, beta=100, init=None):
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
                x_j = X[:,j]
                overlaps[i,j] = np.dot(X[:,i],
                                       np.dot(WTW, X[:,j]))
        return overlaps


    def gradient_search(X, num_steps=200, eta=2*1e-2, plot_all=True):
        # note eta may need to be prop. to beta; 0.1 worked with beta 200
        # performs gradient descent for single basis vector
        # TODO idea for gradient feedback: add terms as basis formed corresponding to 'dot product with basis elements is small'

        err_timeseries = np.zeros(num_steps + 1)
        ALPHA = 1e-3  # lagrange mult for encouraging basis vector separation

        # large local output dir for gradient traj
        outdir_local = outdir + os.sep + 'num_details'
        if not os.path.exists(outdir_local):
            os.makedirs(outdir_local)

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
                    new_xcol = X[:, col] - gradient * eta  # - ALPHA * alternate_obj

                    #magA = np.linalg.norm(gradient)
                    #print('A', magA, eta * magA)
                    #magB = np.linalg.norm(alternate_obj)
                    #print('B', magB, ALPHA * magB)

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
        X = np.random.rand(p, p)*2 - 1  # draw from U(-1,1)
    else:
        assert init.shape == (p, p)
        X = init

    # perform num random searches for basis vector candidates
    x0 = X
    X_final, err_timeseries = gradient_search(x0)
    plot_error_timeseries(err_timeseries, outdir, 'traj')
    plot_error_timeseries(err_timeseries, outdir, 'traj', ylim=(-10, 200))
    WX_final = np.dot(weights, X_final)
    for idx in range(p):
        candidate = WX_final[:, idx]
        plot_basis_candidate_fancy(candidate, idx, outdir, 'final_fancy')
        plot_basis_candidate(candidate, idx, outdir, 'final')
    return X_final


if __name__ == '__main__':
    bigruns = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs'

    # specify dir
    epoch = 0
    runtype = 'hopfield'  # hopfield or normal
    lowdin_approx = False
    X_guess = False
    alt_names = True  # some weights had to be run separately with different naming convention
    hidden_units = 10
    use_fields = False
    maindir = bigruns + os.sep + 'rbm' + os.sep + \
              '%s_%dhidden_%dfields_%.2fbeta_%dbatch_%depochs_%dcdk_%.2Eeta_%dais' % (
                  runtype, hidden_units, use_fields, 2.00, 100, 100, 20, 1e-4, 200)
    run = 0
    if alt_names:
        rundir = maindir
        prepend = '%d_' % run
        weights_ais = 0
    if not alt_names:
        rundir = maindir + os.sep + 'run%d' % run
        prepend = ''
        weights_ais = 200

    # load weights
    obj_title = '%dhidden_%dfields_%dcdk_%dstepsAIS_%.2fbeta' % (hidden_units, use_fields, 20, weights_ais, 2.00)
    weights_obj = np.load(rundir + os.sep + prepend + 'weights_%s.npz' % obj_title)
    weights_timeseries = weights_obj['weights']
    #  (try to) load visible and hidden biases
    assert not use_fields

    # load misc data to get initial transformation guess (R array if hopfield from QR)
    if runtype == 'hopfield' and X_guess:
        from RBM_train import load_rbm_hopfield
        fname = 'hopfield_mnist_%d.npz' % hidden_units
        rbm = load_rbm_hopfield(npzpath=DIR_MODELS + os.sep + 'saved' + os.sep + fname)
        Q, R = rebuild_R_from_xi_image(rbm.xi_image)
        X0_guess = R
    else:
        X0_guess = None

    # choose weights to study
    weights = weights_timeseries[:, :, epoch]
    if lowdin_approx:
        print('Taking Lowdin approx of the weights')
        u, s, vh = np.linalg.svd(weights, full_matrices=False)
        print('Original singular values:\n', s)
        weights = u

    # analysis
    outdir = DIR_OUTPUT + os.sep + 'reversemap' + os.sep + '%s_%dhidden_%dfields_%depoch_%dlowdin' % \
             (runtype, hidden_units, use_fields, epoch, lowdin_approx)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #  binarize_search(weights, outdir, num=10, beta=2000, init=X0_guess)  # OLD WAY --  search vector by vector
    binarize_search_as_matrix(weights, outdir, beta=2000, init=X0_guess)  # NEW WAY - do gradient descent to search for p x p matrix at once
    #plot_weights_timeseries(weights_timeseries, outdir, mode='minmax')
    #plot_weights_timeseries(weights_timeseries, outdir, mode='eval', extra=False)
