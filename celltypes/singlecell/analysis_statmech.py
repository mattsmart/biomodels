import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as clr
from scipy.optimize import fsolve


def params_unpack(params):
    return params['N'], params['beta'], params['r1'], params['r2'], params['kappa1'], params['kappa2']


def params_fill(params_base, pdict):
    params = params_base.copy()
    for k in pdict.keys():
        params[k] = pdict[k]
    return params


def free_energy(m, params):
    N, beta, r1, r2, kappa1, kappa2 = params_unpack(params)
    term0 = (1 - r1 - r2) * np.log(np.cosh(beta * m))
    term1 = r1 * np.log(np.cosh(beta * (m - kappa1)))
    term2 = r2 * np.log(np.cosh(beta * (m + kappa2)))
    return N * m ** 2 / 2 - N/beta * (term0 + term1 + term2)


def free_energy_dm(m, params):
    N, beta, r1, r2, kappa1, kappa2 = params_unpack(params)
    term0 = (1 - r1 - r2) * np.tanh(beta * m)
    term1 = r1 * np.tanh(beta * (m - kappa1))
    term2 = r2 * np.tanh(beta * (m + kappa2))
    return m - (term0 + term1 + term2)


def get_all_roots(params, tol=1e-6):
    mGrid = np.linspace(-1.1, 1.1, 100)
    unique_roots = []
    for mTrial in mGrid:
        solution, infodict, _, _ = fsolve(free_energy_dm, mTrial, args=params, full_output=True)
        mRoot = solution[0]
        append_flag = True
        for k, usol in enumerate(unique_roots):
            if np.abs(mRoot - usol) <= tol:  # only store unique roots from list of all roots
                append_flag = False
                break
        if append_flag:
            if np.linalg.norm(infodict["fvec"]) <= 10e-3:  # only append actual roots (i.e. f(x)=0)
                unique_roots.append(mRoot)
    # remove those which are not stable (keep minima)
    return unique_roots


def is_stable(mval, params, eps=1e-4):
    fval_l = free_energy(mval - eps, params)
    fval = free_energy(mval, params)
    fval_r = free_energy(mval + eps, params)
    return (fval < fval_l and fval < fval_r)


def get_stable_roots(params, tol=1e-6):
    unique_roots = get_all_roots(params, tol=1e-6)
    stable_roots = [mval for mval in unique_roots if is_stable(mval, params)]
    return stable_roots


def plot_f_and_df(params):
    fig, axarr = plt.subplots(1, 2)
    mvals = np.linspace(-2, 2, 100)
    axarr[0].plot(mvals, [free_energy(m, params) for m in mvals])
    axarr[0].set_ylabel(r'$f(m)$')
    axarr[1].plot(mvals, [free_energy_dm(m, params) for m in mvals])
    axarr[1].set_ylabel(r'$df/dm$')
    for idx in xrange(2):
        axarr[idx].set_xlabel(r'$m$')
        axarr[idx].axvline(x=-1, color='k', linestyle='--')
        axarr[idx].axvline(x=1, color='k', linestyle='--')
        axarr[idx].axhline(y=0, color='k', linestyle='-')
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    return


def make_phase_diagram(p1, p2, p1range, p2range, params_base):
    fp_data_2d = np.zeros((len(p1range), len(p2range)), dtype=int)
    for i, p1val in enumerate(p1range):
        print "row", i, "of", len(p1range)
        for j, p2val in enumerate(p2range):
            params = params_fill(params_base, {p1: p1val, p2: p2val})
            fp_data_2d[i,j] = len(get_stable_roots(params, tol=1e-6))

    fs = 16
    # MAKE COLORBAR: https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
    assert np.max(fp_data_2d) <= 4
    cmap = clr.ListedColormap(['lightsteelblue', 'lightgrey', 'thistle', 'moccasin'])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = clr.BoundaryNorm(bounds, cmap.N)

    # regularize values
    # plot image
    fig = plt.figure(figsize=(5,10))
    img = plt.imshow(fp_data_2d, cmap=cmap, interpolation="none", origin='lower', aspect='auto', norm=norm,
                     extent=[p2range[0], p2range[-1], p1range[0], p1range[-1]])
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-')
    ax.set_xlabel(p2, fontsize=fs)
    ax.set_ylabel(p1, fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)

    # add cbar
    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[i-0.5 for i in bounds])

    """
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.savefig(outdir + os.sep + 'fig_fpsum_data_2d_%s_%s_%s.pdf' % (param_1_name, param_2_name, figname_mod),
               bbox_inches='tight')
    """
    plt.show()
    return plt.gca()


if __name__ == '__main__':
    simple_test = False
    phase_diagram = True

    if simple_test:
        params = {
            'N': 100.0,
            'beta': 100.0,
            'N1': 0,
            'N2': 0,
            'kappa1': 0.0,
            'kappa2': 0.0}
        print get_all_roots(params, tol=1e-6)
        print get_stable_roots(params, tol=1e-6)
        plot_f_and_df(params)

    if phase_diagram:
        params = {
            'N': 100.0,
            'beta': 4.0,
            'r1': 0,
            'r2': 0,
            'kappa1': 0.0,
            'kappa2': 0.0}
        p1 = 'kappa1'
        p1range = np.linspace(0.1, 2.0, 30)
        p2 = 'r1'
        p2range = np.linspace(0.0, 0.50, 51)
        make_phase_diagram(p1, p2, p1range, p2range, params)
