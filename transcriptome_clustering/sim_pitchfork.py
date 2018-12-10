import numpy as np
import os
import matplotlib.pyplot as plt

from settings import FOLDER_OUTPUT

"""
TODO
0) rebuild         tArr, xArr, u, v = langevin(hill_coeff, tau, num_slaves, alphas, scale, nCells, timestep, tmax)

1) rebuild         jac, dmat, covM, lyap, langevinData = 
                        flucDiss(tau, hill_coeff, num_slaves, alphas, timestep, tmax, scale, nCells, runcounts)

2) estimate cov properly

3) infer j_ij

4) figure out how to encode changing dynamics here i.e. at some point there is pitchfork bifurcation

"""


def rplus(u, h):
    return np.array([np.divide(1, 1 + u[1, :] ** h, dtype=float), np.divide(1, 1 + u[0, :] ** h, dtype=float)])


def rminus(u, tau):
    return np.array([np.divide(u[0, :], tau, dtype=float), np.divide(u[1, :], tau, dtype=float)])


def sigma(u, dt, h, tau, scale):
    return np.sqrt(dt * (rplus(u, h) + rminus(u, tau)) / scale, dtype=float)


def step(u, dt, h, tau, nExp, scale):
    mean = u + (rplus(u, h) - rminus(u, tau)) * dt
    return np.random.normal(mean, sigma(u, dt, h, tau, scale), np.array([2, nExp]))


def rplusSlave(u, v, h, aVec, N):
    rplusList = []
    for n in range(0, N):
        a = aVec[n]
        rplusList.append(np.divide(a * u[0, :] ** h + 1 - a, 1 + u[0, :] ** h, dtype=float))
    return np.array(rplusList)


def rminusSlave(v, N):
    rminusList = []
    for n in range(0, N):
        rminusList.append(v[n, :])
    return np.array(rminusList)


def sigmaSlave(u, v, aVec, dt, h, N, scale):
    return np.sqrt(dt * (rplusSlave(u, v, h, aVec, N) + rminusSlave(v, N)) / scale, dtype=float)


def stepSlave(u, v, aVec, dt, h, N, nExp, scale):
    mean = v + (rplusSlave(u, v, h, aVec, N) - rminusSlave(v, N)) * dt
    return np.random.normal(mean, sigmaSlave(u, v, aVec, dt, h, N, scale), np.array([N, nExp]))


def getTau(taumin, taumax, dt, tmax, t):
    tauArr1 = np.linspace(taumin, taumax, np.divide(tmax, dt))
    return tauArr1[np.divide(t, dt)]


def getTau1(taumin, taumax, dt, tmax, t):
    k = 10.
    # tArr = np.linspace(0,tmax,np.divide(tmax,dt))
    return taumin + (taumax - taumin) * np.divide(t ** k, t ** k + (tmax / 2.) ** k, dtype=float)


def langevin(hill_coeff, tau, num_slaves, alphas, scale, nCells, timestep, tmax):
    """
    TODO: why is tau sometimes passed as array e.g. in main?
    TODO: what to do with nCells
    TODO: why is last call to langevin(...) in runcounts loop returned as output
    TODO: whats scale
    TODO: whats alphas
    TODO: rename everything better

    Input
    - hill_coeff: model param; pass from flucDiss(...)
    - tau: model param; pass from flucDiss(...)
    - num_slaves: number of non-driver genes in jacobian (which is square matrix of size 2 + num_slave)
    - alphas: ???
    - timestep: pass from flucDiss(...)
    - tmax: pass from flucDiss(...)
    - scale: ???
    - nCells: pass to langevin(...) -- cast as nExpt then unused???
    - runcounts: number of trajectories (ensemble size)

    Returns
    - tArr:
    - xArr:
    - u:
    - v:
    """

    h = hill_coeff
    N = num_slaves
    aVec = alphas  # np.random.uniform(0,1,N)

    nExp = nCells
    dt = timestep  # .01

    # root1 = 1 + .25*(tau-2.) - 0.046875*(tau - 2.)**2 + 0.0136719*(tau - 2.)**3
    root1 = -((2. / 3.) ** (1. / 3.) / (9. * tau + np.sqrt(3.) * np.sqrt(4. + 27. * tau ** 2)) ** (1. / 3.)) + (
                (9 * tau + np.sqrt(3) * np.sqrt(4 + 27 * tau ** 2.)) ** (1. / 3.)) / (2 ** (1. / 3.) * 3 ** (2. / 3.))

    u = root1 * np.ones([2, nExp])
    v = .5 * np.ones([N, nExp])
    t = 0
    gArr = []
    xArr = []
    yArr = []
    zArr = []
    tArr = []

    xArr.append(u[0, 0:10])
    yArr.append(u[1, 0:10])
    zArr.append(v[0, 0:10])
    gArr.append(np.append(u, v, axis=0))
    tArr.append(t)
    count = 0
    while t < tmax:
        print t
        count += 1
        tauT = tau
        v = stepSlave(u, v, aVec, dt, h, N, nExp, scale)
        v = v * (v > 0)
        u = step(u, dt, h, tauT, nExp, scale)
        u = u * (u > 0)
        if count == 1. / dt:
            tArr.append(t)
            xArr.append(u[0, 0:10])
            yArr.append(u[1, 0:10])
            zArr.append(v[0, 0:10])
            gArr.append(np.append(u, v, axis=0))
            count = 0
        t += dt
    return tArr, xArr, u, v

"""
def flucDissOld(tau, hill_coeff, num_slaves, alphas, timestep, tmax, scale, nCells, runcounts):
    h = hill_coeff
    u0ss = -((2. / 3.) ** (1. / 3.) / (9. * tau + np.sqrt(3.) * np.sqrt(4. + 27. * tau ** 2)) ** (1. / 3.)) + \
           ((9 * tau + np.sqrt(3) * np.sqrt(4 + 27 * tau ** 2.)) ** (1. / 3.)) / (2 ** (1. / 3.) * 3 ** (2. / 3.))
    u1ss = u0ss

    jac = np.zeros((2 + num_slaves, 2 + num_slaves))
    jac[0, 0] = -1. / tau
    jac[0, 1] = -h * u1ss ** (h - 1.) / (1 + u1ss ** h) ** 2
    jac[1, 0] = jac[0, 1]
    jac[1, 1] = -1. / tau

    dmat = np.zeros((2 + num_slaves, 2 + num_slaves))
    dmat[0, 0] = np.divide(2. * u0ss / tau, scale, dtype=float)
    dmat[1, 1] = np.divide(2 * u1ss / tau, scale, dtype=float)
    for i in range(num_slaves):
        jac[2 + i, 0] = (h * u0ss ** (h - 1) / (1 + u0ss ** h) ** 2) * (2 * alphas[i] - 1)
        jac[2 + i, 2 + i] = -1
        dmat[2 + i, 2 + i] = np.divide(2 * (alphas[i] * u0ss ** h + 1 - alphas[i]) / (1 + u0ss ** h), scale,
                                       dtype=float)

    runcount = 0
    covM = np.zeros((2 + num_slaves, 2 + num_slaves))
    while runcount < runcounts:
        print 'runcount:', runcount
        runcount += 1
        tArr, xArr, u, v = langevin(hill_coeff, tau, num_slaves, alphas, scale, nCells, timestep, tmax)
        X = np.concatenate((u, v), axis=0)
        covM = np.add(covM, np.cov(X))
    covM = np.divide(covM, runcounts, dtype=float)
    lyap = (np.matmul(jac, covM) + np.matmul(jac, covM).T + dmat) * scale ** 1  # should be zero
    langevinData = tArr, xArr, u, v
    return jac, dmat, covM, lyap, langevinData
"""

def flucDiss(tau, hill_coeff, num_slaves, alphas, timestep, tmax, scale, nCells, runcounts):
    """
    TODO: why is tau sometimes passed as array e.g. in main?
    TODO: what to do with nCells
    TODO: why is last call to langevin(...) in runcounts loop returned as output
    TODO: whats scale
    TODO: whats alphas
    TODO: rename everything better
    TODO: why all these calls to np.divide
    Input
    - tau: model param; pass to langevin(...)
    - hill_coeff: model param; pass to langevin(...)
    - num_slaves: number of non-driver genes in jacobian (which is square matrix of size 2 + num_slave)
    - alphas: ???
    - timestep: pass to langevin(...)
    - tmax: pass to langevin(...)
    - scale: ???
    - nCells: pass to langevin(...) -- cast as nExpt then unused???
    - runcounts: number of trajectories (ensemble size)
    Returns
    - jac: array (2 + num_slaves) x (2 + num_slaves) -- jacobian
    - dmat: array (2 + num_slaves) x (2 + num_slaves) -- diffusion matrix
    - covM: array (2 + num_slaves) x (2 + num_slaves) -- covariance matrix
    - lyap: something relating to lyapunov eqn, which should be ZERO
    - langevinData: tArr, xArr, u, v = langevin(...) -- the output of the last trial in runcounts loop
    """

    h = hill_coeff
    # TODO what are these
    u0ss = -((2. / 3.) ** (1. / 3.) / (9. * tau + np.sqrt(3.) * np.sqrt(4. + 27. * tau ** 2)) ** (1. / 3.)) + \
           ((9 * tau + np.sqrt(3) * np.sqrt(4 + 27 * tau ** 2.)) ** (1. / 3.)) / (2 ** (1. / 3.) * 3 ** (2. / 3.))
    u1ss = u0ss

    jac = np.zeros((2 + num_slaves, 2 + num_slaves))
    jac[0, 0] = -1. / tau
    jac[0, 1] = -h * u1ss ** (h - 1.) / (1 + u1ss ** h) ** 2
    jac[1, 0] = jac[0, 1]
    jac[1, 1] = -1. / tau

    dmat = np.zeros((2 + num_slaves, 2 + num_slaves))
    dmat[0, 0] = np.divide(2. * u0ss / tau, scale, dtype=float)
    dmat[1, 1] = np.divide(2 * u1ss / tau, scale, dtype=float)
    for i in range(num_slaves):
        jac[2 + i, 0] = (h * u0ss ** (h - 1) / (1 + u0ss ** h) ** 2) * (2 * alphas[i] - 1)
        jac[2 + i, 2 + i] = -1
        dmat[2 + i, 2 + i] = np.divide(2 * (alphas[i] * u0ss ** h + 1 - alphas[i]) / (1 + u0ss ** h), scale,
                                       dtype=float)

    runcount = 0
    covM = np.zeros((2 + num_slaves, 2 + num_slaves))
    while runcount < runcounts:
        print 'runcount:', runcount
        runcount += 1
        tArr, xArr, u, v = langevin(hill_coeff, tau, num_slaves, alphas, scale, nCells, timestep, tmax)
        X = np.concatenate((u, v), axis=0)
        covM = np.add(covM, np.cov(X))
    covM = np.divide(covM, runcounts, dtype=float)
    lyap = (np.matmul(jac, covM) + np.matmul(jac, covM).T + dmat) * scale ** 1  # should be zero
    langevinData = tArr, xArr, u, v  # TODO why getting last run of for loop?
    return jac, dmat, covM, lyap, langevinData


"""
def inferCov(jac,dmat):
    len = int(np.sqrt(dmat.size))
    msize = len*(len-1)/2 + len
    A = np.zeros([msize,msize])
    for i in range(len):
        for k in range(i+1):
"""

if __name__ == '__main__':
    hill_coeff = 2.
    num_slaves = 20
    alphas = np.random.uniform(0, 1, num_slaves)
    timestep = .01
    tmax = 2  # 200
    nCells = 10000
    tauArr = [1.9]  # TODO why array
    scale = 1000  # scale for number of proteins in a cell
    runcounts = 1

    for tau in tauArr:
        print 'tau:', tau
        jac, dmat, covM, lyap, langevinData = flucDiss(tau, hill_coeff, num_slaves, alphas, timestep, tmax, scale,
                                                       nCells, runcounts)
        tArr, xArr, u, v = langevinData
        print len(tArr), len(xArr), len(xArr[0]) #(e.g. returns 3, 3, 10)

        plt.plot(tArr, xArr)
        plt.show()

        plt.imshow(covM)
        plt.colorbar()
        savefig = 'covM_scale' + str(scale) + '_runs-' + str(runcounts) + '_ncells-' + str(nCells) + '.png'
        plt.savefig(FOLDER_OUTPUT + os.sep + savefig)
        plt.close()

        plt.imshow(lyap)
        plt.colorbar()
        savefig = 'lyap_scale' + str(scale) + '_runs-' + str(runcounts) + '_ncells-' + str(nCells) + '.png'
        plt.savefig(FOLDER_OUTPUT + os.sep + savefig)
        plt.close()

        plt.hist(u[0, :], alpha=0.5, label=str(tau), density=1)
        plt.legend()
        savefig = 'hist-pitchfork_nCells-' + str(nCells) + '_scale-' + str(scale) + '.png'
        plt.savefig(FOLDER_OUTPUT + os.sep + savefig)
        plt.close()

    # pickle.dump([gArr, tauArr, tArr],open('genes_vs_t_noise_'+str(1./sigmaSupp)+'_taurange_'+str(taumin)+'-'+str(taumax)+'_dt_'+str(dt)+'.p', 'wb'))
