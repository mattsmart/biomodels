import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


def params_unpack(params):
    return params['N'], params['beta'], params['N1'], params['N2'], params['kappa1'], params['kappa2']


def free_energy(m, params):
    N, beta, N1, N2, kappa1, kappa2 = params_unpack(params)
    term0 = (N - N1 - N2) * np.log(np.cosh(beta * m))
    term1 = N1 * np.log(np.cosh(beta * (m - kappa1)))
    term2 = N2 * np.log(np.cosh(beta * (m + kappa2)))
    return N * m ** 2 / 2 - 1/beta * (term0 + term1 + term2)


def free_energy_dm(m, params):
    N, beta, N1, N2, kappa1, kappa2 = params_unpack(params)
    term0 = (N - N1 - N2) * np.tanh(beta * m)
    term1 = N1 * np.tanh(beta * (m - kappa1))
    term2 = N2 * np.tanh(beta * (m + kappa2))
    return m - 1/N * (term0 + term1 + term2)


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


def make_phase_diagram(p1, p2, p1range, p2range, params):
    plt.imshow


if __name__ == '__main__':
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
