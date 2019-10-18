import matplotlib.pyplot as plt
import numpy as np
import os

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, BIFURC_DICT, VALID_BIFURC_PARAMS
from data_io import read_varying_mean_sd_fpt_and_params, write_mfpt_heuristic, read_mfpt_heuristic
from masterqn_approx import linalg_mfpt
from params import Params
from plotting import plot_trajectory_mono, plot_endpoint_mono, plot_table_params
from trajectory import get_centermanifold_traj, trajectory_simulate


def corner_to_flux(corner, params):
    Z_FRACTIONS = {'BL4g': 0.000212,
                   'BR4g': 1.0,
                   'TL4g': 0.000141,
                   'TR4g': 0.507754,
                   'BL100g': 0.0004085,  # note saddle is at 18.98%
                   'TR100g': 0.16514,
                   'BL1g': 0.00020599,
                   'TR1g': 1.0}
    z_fp = Z_FRACTIONS[corner] * params.N  # number entering zhat state per unit time
    MU_1 = 0.0001
    avg_flux = 1/(z_fp * MU_1)
    return avg_flux


def compute_heuristic_mfpt(params):

    # SCRIPT PARAMS
    sim_method = "libcall"  # see constants.py -- sim_methods_valid
    time_start = 0.0
    if params.b == 0.8:
        if params.N < 5*1e4:
            time_end = 3*1600000.0  #20.0
            num_steps = 200000  # number of timesteps in each trajectory
        else:
            time_end = 0.01*1600000.0  #20.0
            num_steps = 200000  # number of timesteps in each trajectory
    else:
        assert params.b == 1.2
        if params.N < 5*1e4:
            time_end = 0.01*1600000.0  #20.0
            num_steps = 200000  # number of timesteps in each trajectory
        else:
            time_end = 0.01*1600000.0  #20.0
            num_steps = 200000  # number of timesteps in each trajectory

    init_cond = [params.N, 0, 0]
    r, times = trajectory_simulate(params, init_cond=init_cond, t0=time_start, t1=time_end, num_steps=num_steps,
                                   sim_method=sim_method)
    """
    for idx in xrange(100):
        x = r[idx,0]
        y = r[idx, 1]
        print idx, r[idx,:], times[idx], (params.a*x + params.b*y + params.c*r[idx,2])/params.N, (params.a*x + params.b*y)/(x+y)
    """

    #plt.plot(times, r[:,2])
    #plt.show()
    # compute integral numerically
    z_of_t = r[:, 2]
    "I = INT_0_inf t N mu z e^(-INT_0_TAU N mu z(t') dt') dt     <- compare vs 1/(mu z_fp)"

    dt = times[1] - times[0]
    expweights = np.zeros(len(times))
    last_expweight = 0
    for idx in xrange(times.shape[0]):
        expweights[idx] = last_expweight
        expweights[idx] += params.mu * dt * z_of_t[idx]
        last_expweight = expweights[idx]
        #weight[i] = params.N * params.mu * dt * np.dot(times[0:idx], z_of_t[0:idx])  big oneline repeat dot

    normalization = 0
    for idx in xrange(times.shape[0]):
        normalization += dt * np.exp(-expweights[idx]) * params.mu * z_of_t[idx]

    mfpt = 0
    for idx in xrange(times.shape[0]):
        mfpt += dt * np.exp(-expweights[idx]) * params.mu * z_of_t[idx] * times[idx]
    mfpt = mfpt / normalization
    print mfpt, normalization
    return mfpt


def plot_heuristic_mfpt(N_range, curve_heuristic, param_vary_name, dataid, show_flag=False, outdir=OUTPUT_DIR, fs=20):
    # load data to compare against
    fnames = {'BL1g': 'mfpt_Nvary_mu1e-4_BL_ens240_xall_g1',
              'BL4g': 'mfpt_Nvary_mu1e-4_BL_ens240_xall_g4',
              'BL100g': 'mfpt_Nvary_mu1e-4_BL_ens240_xall_g100',
              'TR1g': 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g1',
              'TR4g': 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g4',
              'TR100g': 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g100'}
    mfpt_data_dir = 'data' + os.sep + 'mfpt' + os.sep + fnames[dataid]
    mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params = \
        read_varying_mean_sd_fpt_and_params(mfpt_data_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N.txt',
                                            mfpt_data_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N_params.csv')
    if dataid == 'TR1g':
        mfpt_data_dir = 'data' + os.sep + 'mfpt' + os.sep + 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g1_extra'
        mean_fpt_varying_extra, sd_fpt_varying_extra, param_to_vary, param_set, params = \
            read_varying_mean_sd_fpt_and_params(mfpt_data_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N.txt',
                                                mfpt_data_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N_params.csv')
        mean_fpt_varying = mean_fpt_varying + mean_fpt_varying_extra
        sd_fpt_varying = sd_fpt_varying + sd_fpt_varying_extra

    curve_fpflux = [corner_to_flux(dataid, params.mod_copy({'N':n})) for n in N_range]
    write_mfpt_heuristic(N_range, curve_fpflux, filename_mod="_%s_fpFlux" % dataid)

    fit_guess = 0.01
    curve_fit = [1/(params.mu * n * fit_guess) for n in N_range]

    def get_blobtime(n, outer_int_upper=None):
        # TODO use f_arr, s_arr, z_arr, y_arr = get_centermanifold_traj(params)
        # TODO also try with only up to s=0 part
        pmc = params.mod_copy({'N': n})
        blobtime_A = 1
        blobtime_B = 0  # an integral to do

        f_arr, s_arr, z_arr, y_arr = get_centermanifold_traj(pmc, norm=False)

        if outer_int_upper is None:
            outer_int_upper = 1.0 * n

        for i, z in enumerate(z_arr[:-1]):
            if z > outer_int_upper:
                break
            if z >= 1:
                zmid = (z_arr[i + 1] + z_arr[i]) / 2
                dzOuter = z_arr[i + 1] - z_arr[i]
                factor_B_expsum = 0
                for j, z in enumerate(z_arr[:-1]):
                    if z < zmid:
                        smid = (s_arr[j + 1] + s_arr[j]) / 2
                        if smid < 0:
                            factor_B_expsum += 0
                        else:
                            dzInner = z_arr[j + 1] - z_arr[j]
                            factor_B_expsum += smid * dzInner
                        """
                        dzInner = z_arr[j + 1] - z_arr[j]
                        factor_B_expsum += smid * dzInner
                        """
                    else:
                        factor_B_expsum += 0
                        # break
                blobtime_B += 1 / (1 + zmid) * np.exp(factor_B_expsum) * dzOuter
        blobtime = blobtime_A + blobtime_B
        print 'blobtime', n, blobtime_A, blobtime_B
        return blobtime


    def pfix_laplace_blobtime(s, n):
        T = get_blobtime(n, outer_int_upper=None)
        alphaplus = ((2 + s + params.mu) + np.sqrt((2 + s + params.mu) ** 2 - 4 * (1 + s))) / (2 * (1 + s))
        alphaminus = ((2 + s + params.mu) - np.sqrt((2 + s + params.mu) ** 2 - 4 * (1 + s))) / (2 * (1 + s))
        N0 = alphaplus - 1
        N1 = 1 - alphaminus
        expco = (1 + s) * (alphaplus - alphaminus)
        # expansion of den of phi(mu,t) coefficients
        a0 = 1/(N0+N1)
        a1 = (N1 * expco)/(N0+N1)**2
        # pfix coefficients
        c0 = N0 * (alphaminus - alphaplus) * a0 + alphaplus
        c1 = N0 * (alphaminus - alphaplus) * a1
        # option A
        pfix = 1 - c0 - c1 * T
        # option B
        """
        a2 = 1/(N0+N1) * (N1 * expco**2 / (N0+N1) - N1**2 * expco**2 / (N0+N1)**2)
        c2 = N0 * (alphaminus - alphaplus) * a2
        pfix = 1 - c0 - c1 * T - c2 * T**2
        """
        return pfix


    def time_to_hit_boundary(Nval, dual_absorb=False, int_lower=0.0, int_upper=None, init_z=1.0):
        assert int_lower == 0.0
        init_z_normed = init_z / Nval
        pmc = params.mod_copy({'N': Nval})

        f_arr, s_arr, z_arr, y_arr = get_centermanifold_traj(pmc, norm=True)
        num_pts = len(z_arr)

        def A(n, n_idx):
            sval = s_arr[n_idx]
            yval = y_arr[n_idx]
            return sval * n + params.mu * yval

        def B(n, n_idx):
            sval = s_arr[n_idx]
            return (2 + sval) * n / (2 * Nval)  # TODO double check this N^2 not sure

        def psi(n):
            # make sure n and z arr are equivalently normalized or not
            intval = 0.0
            for i, z in enumerate(z_arr[:-1]):
                # TODO integral bounds low high and dz weight
                if z > n:
                    break
                if z > int_lower:
                    zmid = (z_arr[i + 1] + z_arr[i]) / 2
                    dz = z_arr[i + 1] - z_arr[i]
                    intval += A(zmid, i)/B(zmid, i) * dz
            return np.exp(intval)

        def int_one_over_psi(low, high):
            intval = 0.0
            for i, z in enumerate(z_arr[:-1]):
                if z > high:
                    break
                if z >= low:
                    dz = z_arr[i + 1] - z_arr[i]
                    intval += 1 / (psi_table[i]) * dz
            return intval

        time_to_hit_zf = 0.0
        if int_upper is None:
            int_upper = 1.0       # absorbing point, try the unstable height too

        psi_table = np.zeros(num_pts)
        for i, z in enumerate(z_arr[:-1]):
            zmid = (z_arr[i + 1] + z_arr[i]) / 2
            psi_table[i] = psi(zmid)

        if dual_absorb:
            # gardiner p138 eqn 5.2.158
            # compute single integrals
            den = int_one_over_psi(int_lower, int_upper)
            num_A_prefactor = int_one_over_psi(int_lower, init_z_normed)
            num_B_prefactor = int_one_over_psi(init_z_normed, int_upper)
            # compute num A postfactor
            num_A_postfactor = 0
            for i, z in enumerate(z_arr[:-1]):
                if z > int_upper:
                    break
                if z > init_z_normed:
                    zmidOuter = (z_arr[i+1] + z_arr[i]) / 2
                    dzOuter = z_arr[i+1] - z_arr[i]
                    factor_A = 1 / psi_table[i]
                    factor_B_sum = 0
                    for j, z in enumerate(z_arr[:-1]):
                        if z > zmidOuter:
                            break
                        if z > int_lower:
                            zmidInner = (z_arr[j + 1] + z_arr[j]) / 2
                            dzInner = z_arr[j + 1] - z_arr[j]
                            factor_B_sum += psi_table[j] / B(zmidInner, j) * dzInner
                    num_A_postfactor += factor_A * factor_B_sum * dzOuter
            # compute num B postfactor
            num_B_postfactor = 0
            for i, z in enumerate(z_arr[:-1]):
                if z > init_z_normed:
                    break
                if z > int_lower:
                    zmidOuter = (z_arr[i+1] + z_arr[i]) / 2
                    dzOuter = z_arr[i+1] - z_arr[i]
                    factor_A = 1 / psi_table[i]
                    factor_B_sum = 0
                    for j, z in enumerate(z_arr[:-1]):
                        if z > zmidOuter:
                            break
                        if z > int_lower:
                            zmidInner = (z_arr[j + 1] + z_arr[j]) / 2
                            dzInner = z_arr[j + 1] - z_arr[j]
                            factor_B_sum += psi_table[j] / B(zmidInner, j) * dzInner
                        num_B_postfactor += factor_A * factor_B_sum * dzOuter
            # collect terms
            time_to_hit_zf += 1 / den * (num_A_prefactor * num_A_postfactor - num_B_prefactor * num_B_postfactor)
        else:
            time_to_hit_zf = 0
            for i, z in enumerate(z_arr[:-1]):
                if z > int_upper:
                    break
                if z > init_z_normed:
                    zmidOuter = (z_arr[i+1] + z_arr[i]) / 2
                    dzOuter = z_arr[i+1] - z_arr[i]
                    factor_A = 1 / psi_table[i]
                    factor_B_sum = 0
                    for j, z in enumerate(z_arr[:-1]):
                        if z > zmidOuter:
                            break
                        if z > int_lower:
                            zmidInner = (z_arr[j + 1] + z_arr[j]) / 2
                            dzInner = z_arr[j + 1] - z_arr[j]
                            factor_B_sum += psi_table[j] / B(zmidInner, j) * dzInner
                        time_to_hit_zf += factor_A * factor_B_sum * dzOuter
        print 'time_to_hit_boundary BLg100', Nval, time_to_hit_zf
        return time_to_hit_zf

    def prob_to_hit_boundary(Nval, int_lower=0.0, int_upper=1.0, init_z=1.0, hitb=True):
        assert int_lower == 0.0
        init_z_normed = init_z / Nval
        pmc = params.mod_copy({'N': Nval})

        f_arr, s_arr, z_arr, y_arr = get_centermanifold_traj(pmc, norm=True)
        num_pts = len(z_arr)

        def A(n, n_idx):
            sval = s_arr[n_idx]
            yval = y_arr[n_idx]
            return sval * n + params.mu * yval

        def B(n, n_idx):
            sval = s_arr[n_idx]
            return (2 + sval) * n / (2 * Nval)  # TODO double check this N^2 not sure

        def psi(n):
            # make sure n and z arr are equivalently normalized or not
            intval = 0.0
            for i, z in enumerate(z_arr[:-1]):
                # TODO integral bounds low high and dz weight
                if z > n:
                    break
                if z > int_lower:
                    zmid = (z_arr[i + 1] + z_arr[i]) / 2
                    dz = z_arr[i + 1] - z_arr[i]
                    intval += A(zmid, i)/B(zmid, i) * dz
            return np.exp(intval)

        psi_table = np.zeros(num_pts)
        for i, z in enumerate(z_arr[:-1]):
            zmid = (z_arr[i + 1] + z_arr[i]) / 2
            psi_table[i] = psi(zmid)

        def int_psi(low, high):
            intval = 0.0
            for i, z in enumerate(z_arr[:-1]):
                if z > high:
                    break
                if z >= low:
                    dz = z_arr[i + 1] - z_arr[i]
                    intval += 1 / (psi_table[i]) * dz
            return intval

        if hitb:
            prob_exit = int_psi(int_lower, init_z_normed) / int_psi(int_lower, int_upper)
        else:
            prob_exit = int_psi(init_z_normed, int_upper) / int_psi(int_lower, int_upper)
        return prob_exit

    fig = plt.figure()
    ax = plt.gca()

    if dataid == 'TR1g':
        assert params.mult_inc == 1.0 or params.feedback == 'constant'
        fp_stable = np.array([0, 0, 100.0]) / 100.0
        fp_hidden = np.array([80.76849108597227, 19.262827700935464, -0.03131878690773604]) / 100.0

        yfracTRg1 = fp_hidden[1]
        init_avg_div = 1.038  # should be (ax + by)/(x+y)
        s_renorm = (params.c/init_avg_div) - 1
        print "s_renorm", s_renorm
        pfix = s_renorm
        # TRg1 heuristic fixation at all-z
        curve_fit_guess = [1 / (params.mu * n * yfracTRg1 * s_renorm)
                           for n in N_range]
        write_mfpt_heuristic(N_range, curve_fit_guess, filename_mod="_%s_guessPfixTerm1" % dataid)
        curve_fit_guess = [1/(params.mu * n * yfracTRg1 * s_renorm)         # last factor is 1/pfix
                           + np.log(n * s_renorm) / s_renorm
                           + 0.577/s_renorm     # flux from y->z->zhat
                           for n in N_range]
        write_mfpt_heuristic(N_range, curve_fit_guess, filename_mod="_%s_guessPfixTerm123" % dataid)


        N_range_alt = N_range[0:10]
        curve_fit_linalg1 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=False)[0] for n in N_range_alt]
        curve_fit_linalg2 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=True)[0] for n in N_range_alt]
        print 'curve_fit_guess1', curve_fit_linalg1
        print 'curve_fit_guess2', curve_fit_linalg2
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg1, filename_mod="_%s_linalgALLZ" % dataid)
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg2, filename_mod="_%s_linalgZHAT" % dataid)

    elif dataid == 'TR100g':
        assert params.mult_inc == 100.0 and params.feedback != 'constant'
        fp_stable = np.array([41.61623013251053, 41.869771216665875, 16.513998650823595]) / 100.0
        fp_hidden = np.array([71.96914279688974, 28.094892239342407, -0.06403503623214846]) / 100.0

        yfrac_pt0 = fp_hidden[1]
        init_avg_div = 1.056
        zfrac_pt1 = 0.1643  # solve for x y given gamma such that their mean fitness equals z fitness
        yfrac_pt1 = 0.4178
        s_max = 0.0854

        s_renorm = (params.c/init_avg_div) - 1
        print "s_renorm", s_renorm

        curve_fit_guess = [1 / (params.mu * n * yfrac_pt0 * s_renorm)
                           for n in N_range]
        write_mfpt_heuristic(N_range, curve_fit_guess, filename_mod="_%s_guessPfixTerm1" % dataid)
        curve_fit_guess = [1/(params.mu * n * yfrac_pt0 * s_renorm)         # last factor is 1/pfix
                           + np.log(n * s_renorm) / s_renorm
                           + 0.577/s_renorm     # flux from y->z->zhat
                           for n in N_range]
        write_mfpt_heuristic(N_range, curve_fit_guess, filename_mod="_%s_guessPfixTerm123" % dataid)

        # TRg100 heuristic blobtimes v1
        N_range_dense = np.logspace(np.log10(N_range[0]), np.log10(N_range[-5]), 2*len(N_range))
        """
        curve_fit_guess = [1/(params.mu * n * yfrac_pt0 * (1 - np.exp(-params.mu * get_blobtime(n,outer_int_upper=None)**2)))  # last factor is pfix
                           + 0 * 1/(params.mu * n * zfrac_pt1)                                               # direct flux from z1
                           + 0 * 1/(params.mu * n * yfrac_pt1) * 1/(np.sqrt(params.mu * n * s_renorm))       # flux from y->z->zhat
                           for n in N_range_dense]
        """
        # TRg100 heuristic blobtimes v2
        curve_fit_guess = [1/(params.mu * n * yfrac_pt0 * pfix_laplace_blobtime(s_renorm, n))
                           for n in N_range_dense]
        write_mfpt_heuristic(N_range_dense, curve_fit_guess, filename_mod="_%s_guessBlobtimesLaplace" % dataid)

        """curve_fit_guess = [1 / (params.mu**2 * n**2 * zfrac_pt1)
                   for n in N_range]"""
        #vertlne = 1/(zfrac_pt1 * s_renorm)   # when N = 1/(s0z0)
        #ax.axvline(vertlne)

        N_range_alt = N_range[0:10]
        curve_fit_linalg1 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=False)[0] for n in N_range_alt]
        curve_fit_linalg2 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=True)[0] for n in N_range_alt]
        print 'curve_fit_guess1', curve_fit_linalg1
        print 'curve_fit_guess2', curve_fit_linalg2
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg1, filename_mod="_%s_linalgALLZ" % dataid)
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg2, filename_mod="_%s_linalgZHAT" % dataid)

        # mfpt FPE heuristic (TODO how to combine these))
        """
        N_range_timeBoundary = N_range[0:7]  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guessTime1 = [1 / (params.mu * n * yfrac_pt0 *
                                 time_to_hit_boundary(n, dual_absorb=False, int_lower=0.0, int_upper=fp_stable[2], init_z=1.0))
                                 for n in N_range_timeBoundary]
        curve_fit_guessTime2 = [1 / (params.mu * n * yfrac_pt0 *
                                 time_to_hit_boundary(n, dual_absorb=False, int_lower=0.0, int_upper=None, init_z=1.0))
                                 for n in N_range_timeBoundary]
        curve_fit_guessTime3 = [1 / (params.mu * n * yfrac_pt0 *
                                 time_to_hit_boundary(n, dual_absorb=True, int_lower=0.0, int_upper=fp_stable[2], init_z=1.0))
                                 for n in N_range_timeBoundary]
        curve_fit_guessTime4 = [1 / (params.mu * n * yfrac_pt0 *
                                 time_to_hit_boundary(n, dual_absorb=True, int_lower=0.0, int_upper=None, init_z=1.0))
                                 for n in N_range_timeBoundary]
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guessTime1, filename_mod="_%s_guessBoundaryTimeMono1" % dataid)
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guessTime2, filename_mod="_%s_guessBoundaryTimeMono2" % dataid)
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guessTime3, filename_mod="_%s_guessBoundaryTimeDual1" % dataid)
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guessTime4, filename_mod="_%s_guessBoundaryTimeDual2" % dataid)
        """
        # prob hit boundary heuristic
        N_range_probBoundary = N_range[0:7]  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guessProb1 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=fp_stable[2], init_z=1.0))
                                 for n in N_range]
        curve_fit_guessProb2 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=0.5, init_z=1.0))
                                 for n in N_range_probBoundary]
        curve_fit_guessProb3 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=1.0, init_z=1.0))
                                 for n in N_range_probBoundary]
        write_mfpt_heuristic(N_range, curve_fit_guessProb1, filename_mod="_%s_guessBoundaryProb1" % dataid)
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guessProb2, filename_mod="_%s_guessBoundaryProb2" % dataid)
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guessProb3, filename_mod="_%s_guessBoundaryProb3" % dataid)

    elif dataid == 'TR4g':
        assert params.mult_inc == 4.0 and params.feedback != 'constant'
        fp_stable = np.array([24.588047090562984, 24.63656778168078, 50.77538512775624]) / 100.0
        fp_hidden = np.array([80.37960284434968, 19.652759713453463, -0.03236255780313968]) / 100.0

        N_range_alt = N_range[0:10]
        curve_fit_linalg1 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=False)[0] for n in N_range_alt]
        curve_fit_linalg2 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=True)[0] for n in N_range_alt]
        print 'curve_fit_guess1', curve_fit_linalg1
        print 'curve_fit_guess2', curve_fit_linalg2
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg1, filename_mod="_%s_linalgALLZ" % dataid)
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg2, filename_mod="_%s_linalgZHAT" % dataid)

    elif dataid == 'BL100g':
        assert params.mult_inc == 100.0 and params.feedback != 'constant'
        fp_low = np.array([77.48756569595079, 22.471588735222426, 0.04084556882678214]) / 100.0
        fp_mid = np.array([40.61475564788107, 40.401927055159106, 18.983317296959825]) / 100.0
        yfrac_pt0 = fp_low[1]
        """
        init_avg_div = 1.056  # TODO
        zfrac_pt1 = 0.1643  # TODO solve for x y given gamma such that their mean fitness equals z fitness
        yfrac_pt1 = 0.4178  # TODO 
        s_max = 0.0854      # TODO
        """

        # blobtime heuristic
        N_range_alt = N_range  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guess1 = [1 / (params.mu * n * yfrac_pt0 * (
                1 - np.exp(-params.mu * get_blobtime(n, outer_int_upper=n * fp_mid[2]) ** 2)))
                           for n in N_range_alt]
        curve_fit_guess2 = [1 / (params.mu * n * yfrac_pt0 * (
                1 - np.exp(-params.mu * get_blobtime(n, outer_int_upper=n * 0.4)) ** 2))
                            for n in N_range_alt]
        curve_fit_guess3 = [1 / (params.mu * n * yfrac_pt0 * (
                1 - np.exp(-params.mu * get_blobtime(n, outer_int_upper=None) ** 2)))
                            for n in N_range_alt]
        write_mfpt_heuristic(N_range_alt, curve_fit_guess1, filename_mod="_%s_guessBlobtimes1" % dataid)
        write_mfpt_heuristic(N_range_alt, curve_fit_guess2, filename_mod="_%s_guessBlobtimes2" % dataid)
        write_mfpt_heuristic(N_range_alt, curve_fit_guess3, filename_mod="_%s_guessBlobtimes3" % dataid)

        # mfpt FPE heuristic (TODO how to combine these))
        N_range_timeBoundary = N_range  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guess1 = [1 / (params.mu * n * yfrac_pt0 *
                                 time_to_hit_boundary(n, dual_absorb=False, int_lower=0.0, int_upper=fp_mid[2], init_z=1.0))
                                 for n in N_range_timeBoundary]
        curve_fit_guess2 = [1 / (params.mu * n * yfrac_pt0 *
                                 time_to_hit_boundary(n, dual_absorb=False, int_lower=0.0, int_upper=None, init_z=1.0))
                                 for n in N_range_timeBoundary]
        curve_fit_guess3 = [1 / (params.mu * n * yfrac_pt0 *
                                 time_to_hit_boundary(n, dual_absorb=True, int_lower=0.0, int_upper=fp_mid[2], init_z=1.0))
                                 for n in N_range_timeBoundary]
        curve_fit_guess4 = [1 / (params.mu * n * yfrac_pt0 *
                                 time_to_hit_boundary(n, dual_absorb=True, int_lower=0.0, int_upper=None, init_z=1.0))
                                 for n in N_range_timeBoundary]
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guess1, filename_mod="_%s_guessBoundaryTimeMono1" % dataid)
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guess2, filename_mod="_%s_guessBoundaryTimeMono2" % dataid)
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guess3, filename_mod="_%s_guessBoundaryTimeDual1" % dataid)
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guess4, filename_mod="_%s_guessBoundaryTimeDual2" % dataid)

        # MASTER EQN SOLVE BLOCK
        N_range_alt = N_range[0:10]
        curve_fit_linalg1 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=False)[0] for n in N_range_alt]
        curve_fit_linalg2 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=True)[0] for n in N_range_alt]
        print 'curve_fit_guess1', curve_fit_linalg1
        print 'curve_fit_guess2', curve_fit_linalg2
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg1, filename_mod="_%s_linalgALLZ" % dataid)
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg2, filename_mod="_%s_linalgZHAT" % dataid)

        # prob hit boundary heuristic
        N_range_probBoundary = N_range  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guess1 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=fp_mid[2], init_z=1.0))
                                 for n in N_range_probBoundary]
        curve_fit_guess2 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=0.4, init_z=1.0))
                                 for n in N_range_probBoundary]
        curve_fit_guess3 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=1.0, init_z=1.0))
                                 for n in N_range_probBoundary]
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess1, filename_mod="_%s_guessBoundaryProb1" % dataid)
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess2, filename_mod="_%s_guessBoundaryProb2" % dataid)
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess3, filename_mod="_%s_guessBoundaryProb3" % dataid)

    elif dataid == 'BL4g':
        assert params.mult_inc == 4.0 and params.feedback != 'constant'
        fp_low = np.array([85.08181731871301, 14.89695736662967, 0.02122531465732358]) / 100.0
        fp_mid = np.array([21.56844087406341, 21.53060213939143, 56.900956986545154]) / 100.0
        yfrac_pt0 = fp_low[1]
        # MASTER EQN SOLVE BLOCK
        N_range_alt = N_range[0:10]
        curve_fit_linalg1 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=False)[0] for n in N_range_alt]
        curve_fit_linalg2 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=True)[0] for n in N_range_alt]
        print 'curve_fit_guess1', curve_fit_linalg1
        print 'curve_fit_guess2', curve_fit_linalg2
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg1, filename_mod="_%s_linalgALLZ" % dataid)
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg2, filename_mod="_%s_linalgZHAT" % dataid)

        # prob hit boundary heuristic
        N_range_probBoundary = N_range  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guess1 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=fp_mid[2], init_z=1.0))
                                 for n in N_range_probBoundary]
        curve_fit_guess2 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=0.8, init_z=1.0))
                                 for n in N_range_probBoundary]
        curve_fit_guess3 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=1.0, init_z=1.0))
                                 for n in N_range_probBoundary]
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess1, filename_mod="_%s_guessBoundaryProb1" % dataid)
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess2, filename_mod="_%s_guessBoundaryProb2" % dataid)
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess3, filename_mod="_%s_guessBoundaryProb3" % dataid)

    elif dataid == 'BL1g':
        print params.mult_inc, params.mult_dec
        assert params.mult_inc == 1.0 or params.feedback == 'constant'
        fp_low = np.array([85.39353129821689, 14.585869420527702, 0.02059928125541255]) / 100.0
        fp_mid = np.array([0,0,100.0]) / 100.0
        yfrac_pt0 = fp_low[1]
        # MASTER EQN SOLVE BLOCK
        N_range_alt = N_range[0:10]
        curve_fit_linalg1 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=False)[0] for n in N_range_alt]
        curve_fit_linalg2 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=True)[0] for n in N_range_alt]
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg1, filename_mod="_%s_linalgALLZ" % dataid)
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg2, filename_mod="_%s_linalgZHAT" % dataid)

        # prob hit boundary heuristic
        N_range_probBoundary = N_range  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guess1 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=fp_mid[2], init_z=1.0))
                                 for n in N_range_probBoundary]
        curve_fit_guess2 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=0.8, init_z=1.0))
                                 for n in N_range_probBoundary]
        curve_fit_guess3 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=1.0, init_z=1.0))
                                 for n in N_range_probBoundary]
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess1, filename_mod="_%s_guessBoundaryProb1" % dataid)
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess2, filename_mod="_%s_guessBoundaryProb2" % dataid)
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess3, filename_mod="_%s_guessBoundaryProb3" % dataid)

    else:
        curve_fit_guess = [0 for n in N_range]
        print 'no fit guess for %s' % dataid

    plt.plot(N_range, curve_fpflux, '--k', label='curve_fpflux')
    plt.plot(N_range, curve_heuristic, '-or', label='curve_heuristic')
    plt.plot(N_range[:len(mean_fpt_varying)], mean_fpt_varying, '-ok', label='data')
    #plt.plot(N_range, curve_fit, '--b', label=r'fit $1/(a \mu N), a=%.2f$' % fit_guess)
    plt.plot(N_range_dense, curve_fit_guess, '--b', label=r'blobtime laplace')
    
    """
    # prob absorb b end block
    plt.plot(N_range_alt, curve_fit_guess1, '--g', label=r'prob b zmax ~$0.2$ ($z_{us}$)')
    plt.plot(N_range_alt, curve_fit_guess2, '--b', label=r'prob b zmax $0.4$')
    plt.plot(N_range_alt, curve_fit_guess3, '--p', label=r'prob b zmax $1.0$')
    #plt.plot(N_range_dense, curve_fit_guess4, '-.b', label=r'dual zmax $1.0$')
    """

    # MASTER EQN BLOCK
    plt.plot(N_range_alt, curve_fit_linalg1, '--b', label=r'1D ME mfpt all-z')
    plt.plot(N_range_alt, curve_fit_linalg2, '--g', label=r'1D ME mfpt zhat')

    ax = plt.gca()
    ax.set_xlabel(r'$%s$' % param_vary_name, fontsize=fs)
    ax.set_ylabel(r'$\tau$', fontsize=fs)
    #ax.set_ylabel(r'$\langle\tau\rangle$', fontsize=fs)
    #ax.set_ylabel(r'$\delta\tau$', fontsize=fs)

    plt.xticks(fontsize=fs-2)
    plt.yticks(fontsize=fs-2)

    # log options
    flag_xlog10 = True
    flag_ylog10 = True
    if flag_xlog10:
        #ax.set_xscale("log", nonposx='clip')
        ax.set_xscale("log")
        #ax_dual.set_xscale("log", nonposx='clip')
        #ax.set_xlim([np.min(param_set)*0.9, 1.5*1e4])
    if flag_ylog10:
        #ax.set_yscale("log", nonposx='clip')
        ax.set_yscale("log")
        #ax_dual.set_yscale("log", nonposx='clip')
        ax.set_ylim([6*1e-1, 3*1e6])

    ax.legend(fontsize=fs-6, ncol=2, loc='upper right')
    plt_save = "mean_fpt_varying_heuristic_" + dataid
    plt.savefig(outdir + os.sep + plt_save + '.pdf', bbox_inches='tight')
    if show_flag:
        plt.show()
    return ax


if __name__ == '__main__':
    param_varying_name = "N"
    assert param_varying_name == "N"

    # DYNAMICS PARAMETERS
    system = "feedback_z"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
    feedback = "tanh"  # "constant", "hill", "step", "pwlinear", "tanh"
    params_dict = {
        'alpha_plus': 0.2,
        'alpha_minus': 1.0,  # 0.5
        'mu': 1e-4,  # 0.01
        'a': 1.0,
        'b': 0.8,
        'c': 0.9,  # 1.2
        'N': 100.0,  # 100.0
        'v_x': 0.0,
        'v_y': 0.0,
        'v_z': 0.0,
        'mu_base': 0.0,
        'c2': 0.0,
        'v_z2': 0.0,
        'mult_inc': 100.0,
        'mult_dec': 100.0,
    }
    params = Params(params_dict, system, feedback=feedback)
    data_id = 'TR100g'

    N_range = [int(a) for a in np.logspace(1.50515, 4.13159, num=11)] + [int(a) for a in np.logspace(4.8, 7, num=4)]

    # OTHER PARAMETERS
    #init_cond = np.zeros(params.numstates, dtype=int)
    #init_cond[0] = int(params.N)
    curve_heuristic = [0]*len(N_range)
    for idx, N in enumerate(N_range):
        pv = params.mod_copy({'N': N})
        curve_heuristic[idx] = compute_heuristic_mfpt(pv)
        print N, curve_heuristic[idx]
    write_mfpt_heuristic(N_range, curve_heuristic, filename_mod="_%s_fpRouteFlux" % data_id)

    plot_heuristic_mfpt(N_range, curve_heuristic, 'N', data_id, fs=20)
