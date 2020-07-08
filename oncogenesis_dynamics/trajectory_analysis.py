import matplotlib.pyplot as plt
import numpy as np
import os

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, BIFURC_DICT, VALID_BIFURC_PARAMS
from data_io import read_varying_mean_sd_fpt_and_params, write_mfpt_heuristic, read_mfpt_heuristic
from masterqn_approx import linalg_mfpt
from params import Params
from presets import presets
from plotting import plot_trajectory_mono, plot_endpoint_mono, plot_table_params
from trajectory import get_centermanifold_traj, trajectory_simulate


fnames = {'BL1g': 'mfpt_Nvary_mu1e-4_BL_ens240_xall_g1',
          'BL4g': 'mfpt_Nvary_mu1e-4_BL_ens240_xall_g4',
          'BL100g': 'mfpt_Nvary_mu1e-4_BL_ens240_xall_g100',
          'TR1g': 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g1',
          'TR4g': 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g4',
          'TR100g': 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g100'}


def corner_to_flux(corner, params):
    Z_FRACTIONS = {'BL4g': 0.000212,
                   'BR4g': 1.0,
                   'TL4g': 0.000141,
                   'TR4g': 0.507754,
                   'BL100g': 0.0004085,  # note saddle is at 18.98%
                   'TR100g': 0.16514,
                   'BL1g': 0.00020599,
                   'TR1g': 1.0}
    if corner in fnames:
        z_fp = Z_FRACTIONS[corner] * params.N  # number entering zhat state per unit time
        MU_1 = 0.0001
        avg_flux = 1/(z_fp * MU_1)
    else:
        avg_flux = 0
    return avg_flux


def compute_integral_path_mfpt(params):
    # computes the integral expression tau_P

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
                                   sim_method=sim_method, fp_comparison=False)
    """
    for idx in xrange(100):
        x = r[idx,0]
        y = r[idx, 1]
        print idx, r[idx,:], times[idx], (params.a*x + params.b*y + params.c*r[idx,2])/params.N, (params.a*x + params.b*y)/(x+y)
    """
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


def generate_heuristic_mfpt(params, N_range, param_vary_name, dataid, explicit=False, skip_linalg=False, alt_absorb=False):

    # Curve set #1 -- write 1 / mu z*
    curve_fpflux = [corner_to_flux(dataid, params.mod_copy({'N': n})) for n in N_range]
    write_mfpt_heuristic(N_range, curve_fpflux, filename_mod="_%s_fpFlux" % dataid)

    # Curve set #2 -- refinement to #1 is the tau_P integral
    print "Computing tau_P..."
    """N_range_extend = [int(a) for a in np.logspace(1.50515, 4.13159, num=11)] + [int(a) for a in np.logspace(4.8, 7, num=4)]
    curve_tau_P = [compute_integral_path_mfpt(params.mod_copy({'N': n})) for n in N_range_extend]
    write_mfpt_heuristic(N_range_extend, curve_tau_P, filename_mod="_%s_fpRouteFlux" % dataid)"""

    # Curve set #3 -- direct master equation objects, with or without the explicit form of the rates
    explicit_fname = {True: 'Expl', False: ''}
    if not skip_linalg:
        print "Computing the three master equation MFPTs ..."
        N_range_alt_short = N_range#[0:8]
        N_range_alt_long = N_range#[0:10]
        #curve_fit_linalg1 = [linalg_mfpt(params=params.mod_copy({'N': n}), explicit=explicit, flag_zhat=False, use_eval=True) for n in N_range_alt_short]
        curve_fit_linalg2 = [linalg_mfpt(params=params.mod_copy({'N': n}), explicit=explicit, flag_zhat=False) for n in N_range_alt_short]
        curve_fit_linalg3 = [linalg_mfpt(params=params.mod_copy({'N': n}), explicit=explicit, flag_zhat=True) for n in N_range_alt_long]
        #write_mfpt_heuristic(N_range_alt_short, curve_fit_linalg1, filename_mod="_%s_linalgALLZeval%s" % (dataid, explicit_fname[explicit]))
        write_mfpt_heuristic(N_range_alt_short, curve_fit_linalg2, filename_mod="_%s_linalgALLZ%s%s" % (dataid, explicit_fname[explicit]))
        write_mfpt_heuristic(N_range_alt_long, curve_fit_linalg3, filename_mod="_%s_linalgZHAT%s%s" % (dataid, explicit_fname[explicit]))

        if alt_absorb:
            assert dataid == 'TR100g'
            alt_absorb_fraction = 0.16513998
            curve_fit_linalg_altabsorb = [linalg_mfpt(params=params.mod_copy({'N': n}), explicit=explicit, flag_zhat=False,
                                                       alt_absorb_point=int(alt_absorb_fraction * n)) for n in N_range]
            write_mfpt_heuristic(N_range, curve_fit_linalg_altabsorb, filename_mod="_%s_linalgALLZ%sAltAbsorb" % (dataid, explicit_fname[explicit]))

    def A(zval, yval, f_val, s_val, explicit=True):
        # birth rate minus the death rate
        if explicit:
            Aval = (params.c - f_val) * zval + params.mu * yval
        else:
            Aval = s_val * zval + params.mu * yval
        return Aval

    def B(zval, yval, f_val, s_val, Nval, explicit=True):
        # birth rate plus the death rate, then scaled by 1/2N
        if explicit:
            Bval = ((params.c + f_val) * zval + params.mu * yval) / 2.0 #(2.0 * Nval)  # (2 * z_normed * Nval)
            #Bval = ((params.c + f_val) * z_normed + params.mu * y_normed) / 2
        else:
            Bval = (2 + s_val) * zval / (2 * Nval)
        return Bval

    def get_blobtime_archive(n, outer_int_upper=None):
        # TODO normalization care
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
                        assert 1 == 2  # TODO use expplicit form of birth death rates
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

    def get_blobtime(n, outer_int_upper=None, explicit=True):
        # TODO normalization care
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
                if explicit:
                    blobtime_B += 1 / (1 + zmid) * f_arr[i+1] * np.exp(factor_B_expsum) * dzOuter  # note the f array indexing is for z, not z+1 as we ne need
                else:
                    blobtime_B += 1 / (1 + zmid) * np.exp(factor_B_expsum) * dzOuter
        blobtime = blobtime_A + blobtime_B
        print 'blobtime', n, blobtime_A, blobtime_B
        return blobtime

    def pfix_laplace_blobtime(s, n):
        T = get_blobtime(n, outer_int_upper=None)
        alphaplus = ((2 + s + params.mu) + np.sqrt((2 + s + params.mu) ** 2 - 4 * (1 + s))) / (2 * (1 + s))
        alphaminus = ((2 + s + params.mu) - np.sqrt((2 + s + params.mu) ** 2 - 4 * (1 + s))) / (2 * (1 + s))
        assert 1 == 2  # TODO use expplicit form of birth death rates
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

    def time_to_hit_boundary_preMay8(Nval, dual_absorb=False, int_lower=0.0, int_upper=None, init_z=1.0,
                             troubleshoot_slowmanifold=False):
        assert int_lower == 0.0
        pmc = params.mod_copy({'N': Nval})

        # Note: not normed
        f_arr, s_arr, z_arr, y_arr = get_centermanifold_traj(pmc, norm=True)
        num_pts = len(z_arr)

        time_to_hit_zf = 0.0
        if int_upper is None:
            int_upper = 1.0       # absorbing point, try the unstable height as well as z = Nval

        A_table = np.zeros(num_pts - 1)
        B_table = np.zeros(num_pts - 1)
        expval_table = np.zeros(num_pts - 1)
        expval = 0.0
        for i, z in enumerate(z_arr[:-1]):
            dz = z_arr[i + 1] - z_arr[i]
            zmid = (z_arr[i + 1] + z_arr[i]) / 2
            A_table[i] = A(z_arr[i], y_arr[i], f_arr[i], s_arr[i])  # TODO consider zmid and all other quantities mid as well
            B_table[i] = B(z_arr[i], y_arr[i], f_arr[i], s_arr[i], Nval)
            #print "UPPER PRINT: i, z, z_arr[i], y_arr[i], expval, A_table[i], B_table[i]"
            #print i, z, z_arr[i], y_arr[i], expval, A_table[i], B_table[i]
            if int_lower < zmid:
                expval += A_table[i] / B_table[i] * dz
                expval_table[i] = expval             # psi is exp of this, note gardiner 5.2.157, here B already has "1/2"
            #print i, z, A_table[i], B_table[i], expval, dz
        psi_table = np.exp(expval_table)

        def int_one_over_psi(low, high):
            intval = 0.0
            for i, z in enumerate(z_arr[:-1]):
                if z > high:
                    break
                if z >= low:
                    dz = z_arr[i + 1] - z_arr[i]
                    intval += 1 / (psi_table[i]) * dz
            return intval

        if troubleshoot_slowmanifold:
            plt.plot(z_arr, label='z')
            plt.plot(y_arr, label='y')
            plt.plot(1.0 - z_arr - y_arr, label='x')
            plt.legend()
            plt.show()

            plt.plot(f_arr, label='f')
            plt.plot(s_arr, label='s')
            plt.legend()
            plt.show()

            plt.plot(A_table, label='A')
            plt.plot(B_table, label='B')
            plt.legend()
            plt.show()

            plt.plot(z_arr[:-1], A_table, label='A(z)')
            plt.plot(z_arr[:-1], B_table, label='B(z)')
            plt.legend()
            plt.show()

            plt.plot(expval_table, label=r'$\ln \psi$')
            plt.legend()
            plt.show()

            plt.plot(z_arr[:-1], expval_table, label=r'$\ln \psi(z)$')
            plt.legend()
            plt.show()

        if dual_absorb:
            # gardiner p138 eqn 5.2.158
            # compute single integrals
            den = int_one_over_psi(int_lower, int_upper)
            num_A_prefactor = int_one_over_psi(int_lower, init_z)
            num_B_prefactor = int_one_over_psi(init_z, int_upper)
            # compute num A postfactor
            num_A_postfactor = 0
            for i, z in enumerate(z_arr[:-1]):
                if z > int_upper:
                    break
                if z > init_z:
                    zmidOuter = (z_arr[i+1] + z_arr[i]) / 2
                    dzOuter = z_arr[i+1] - z_arr[i]
                    factor_B_sum = 0
                    for j, z in enumerate(z_arr[:-1]):
                        if z > zmidOuter:
                            break
                        if z > int_lower:
                            dzInner = z_arr[j + 1] - z_arr[j]
                            #factor_B_sum += psi_table[j] / B_table[j] * dzInner
                            expval_diff = expval_table[j] - expval_table[i]
                            factor_B_sum += np.exp( expval_diff ) / B_table[j] * dzInner
                num_A_postfactor += factor_B_sum * dzOuter
            # compute num B postfactor
            num_B_postfactor = 0
            for i, z in enumerate(z_arr[:-1]):
                if z > init_z:
                    break
                if z > int_lower:
                    zmidOuter = (z_arr[i + 1] + z_arr[i]) / 2
                    dzOuter = z_arr[i + 1] - z_arr[i]
                    factor_B_sum = 0
                    for j, z in enumerate(z_arr[:-1]):
                        if z > zmidOuter:
                            break
                        if z > int_lower:
                            dzInner = z_arr[j + 1] - z_arr[j]
                            expval_diff = expval_table[j] - expval_table[i]
                            factor_B_sum += np.exp(expval_diff) / B_table[j] * dzInner
                num_B_postfactor += factor_B_sum * dzOuter
            # collect terms
            time_to_hit_zf = 1 / den * (num_A_prefactor * num_A_postfactor - num_B_prefactor * num_B_postfactor)
        else:
            # gardiner p139, eqn 5.2.160
            time_to_hit_zf = 0
            for i, z in enumerate(z_arr[:-1]):
                if z > int_upper:
                    break
                if z > init_z:
                    zmidOuter = (z_arr[i+1] + z_arr[i]) / 2
                    dzOuter = z_arr[i+1] - z_arr[i]
                    factor_B_sum = 0
                    for j, z in enumerate(z_arr[:-1]):
                        if z > zmidOuter:
                            break
                        if z > int_lower:
                            dzInner = z_arr[j + 1] - z_arr[j]
                            #expval_diff = expval_table[j] - expval_table[i]
                            factor_B_sum += np.exp(expval_table[j]) / B_table[j] * dzInner
                    time_to_hit_zf += factor_B_sum * dzOuter / np.exp(expval_table[i])
        print 'time_to_hit_boundary for Nval', Nval, 'is', time_to_hit_zf
        return time_to_hit_zf

    def time_to_hit_boundary(Nval, dual_absorb=False, int_lower=0.0, int_upper=None, init_z=1.0,
                             troubleshoot_slowmanifold=False):
        pmc = params.mod_copy({'N': Nval})

        NORMED = False
        #assert NORMED == True
        if NORMED:
            int_lower = int_lower / float(Nval)
            init_z = init_z / float(Nval)

        # Note: not normed
        f_arr, s_arr, z_arr, y_arr = get_centermanifold_traj(pmc, norm=NORMED)
        num_pts = len(z_arr)

        time_to_hit_zf = 0.0
        if int_upper is None:
            int_upper = Nval       # absorbing point, try the unstable height as well as z = Nval
            if NORMED:
                int_upper = 1.0
        if NORMED and int_upper > 1.0:
            print "warning, normalizing int_upper in time_to_hit_boundary"
            int_upper = int_upper / Nval

        #print 'time_to_hit_boundary', Nval, 'upper:', int_upper

        A_table = np.zeros(num_pts - 1)
        B_table = np.zeros(num_pts - 1)
        expval_table = np.zeros(num_pts - 1)
        expval = 0.0
        for i, z in enumerate(z_arr[:-1]):
            dz = z_arr[i + 1] - z_arr[i]
            zmid = (z_arr[i + 1] + z_arr[i]) / 2
            A_table[i] = A(z_arr[i], y_arr[i], f_arr[i], s_arr[i])  # TODO consider zmid and all other quantities mid as well
            B_table[i] = B(z_arr[i], y_arr[i], f_arr[i], s_arr[i], Nval)
            # print "UPPER PRINT: i, z, z_arr[i], y_arr[i], expval, A_table[i], B_table[i]"
            # print i, z, z_arr[i], y_arr[i], expval, A_table[i], B_table[i]
            if int_lower < zmid:
                expval += A_table[i] / B_table[i] * dz
                expval_table[i] = expval       # psi is exp of this, note gardiner 5.2.157, here B already has "1/2"
            # print i, z, A_table[i], B_table[i], expval, dz
        psi_table = np.exp(expval_table)

        def int_one_over_psi(low, high):
            intval = 0.0
            for i, z in enumerate(z_arr[:-1]):
                if z > high:
                    break
                if z >= low:
                    dz = z_arr[i + 1] - z_arr[i]
                    intval += 1 / (psi_table[i]) * dz
            return intval

        if troubleshoot_slowmanifold:
            plt.plot(z_arr, label='z')
            plt.plot(y_arr, label='y')
            if NORMED:
                plt.plot(1.0 - z_arr - y_arr, label='x')
            else:
                plt.plot(Nval - z_arr - y_arr, label='x')
            plt.legend()
            plt.show()

            plt.plot(f_arr, label='f')
            plt.plot(s_arr, label='s')
            plt.legend()
            plt.show()

            plt.plot(A_table, label='A')
            plt.plot(B_table, label='B')
            plt.legend()
            plt.show()

            plt.plot(z_arr[:-1], A_table, label='A(z)')
            plt.plot(z_arr[:-1], B_table, label='B(z)')
            plt.legend()
            plt.show()

            plt.plot(expval_table, label=r'$\ln \psi$')
            plt.legend()
            plt.show()

            plt.plot(z_arr[:-1], expval_table, label=r'$\ln \psi(z)$')
            plt.legend()
            plt.show()

        if dual_absorb:
            # gardiner p138 eqn 5.2.158
            # compute single integrals
            den = int_one_over_psi(int_lower, int_upper)
            num_A_prefactor = int_one_over_psi(int_lower, init_z)
            num_B_prefactor = int_one_over_psi(init_z, int_upper)
            # compute num A postfactor
            num_A_postfactor = 0
            for i, z in enumerate(z_arr[:-1]):
                if z > int_upper:
                    break
                if z > init_z:
                    zmidOuter = (z_arr[i+1] + z_arr[i]) / 2
                    dzOuter = z_arr[i+1] - z_arr[i]
                    factor_B_sum = 0
                    for j, z in enumerate(z_arr[:-1]):
                        if z > zmidOuter:
                            break
                        if z > int_lower:
                            dzInner = z_arr[j + 1] - z_arr[j]
                            #factor_B_sum += psi_table[j] / B_table[j] * dzInner
                            expval_diff = expval_table[j] - expval_table[i]
                            factor_B_sum += np.exp( expval_diff ) / B_table[j] * dzInner
                num_A_postfactor += factor_B_sum * dzOuter
            # compute num B postfactor
            num_B_postfactor = 0
            for i, z in enumerate(z_arr[:-1]):
                if z > init_z:
                    break
                if z > int_lower:
                    zmidOuter = (z_arr[i + 1] + z_arr[i]) / 2
                    dzOuter = z_arr[i + 1] - z_arr[i]
                    factor_B_sum = 0
                    for j, z in enumerate(z_arr[:-1]):
                        if z > zmidOuter:
                            break
                        if z > int_lower:
                            dzInner = z_arr[j + 1] - z_arr[j]
                            expval_diff = expval_table[j] - expval_table[i]
                            factor_B_sum += np.exp(expval_diff) / B_table[j] * dzInner
                num_B_postfactor += factor_B_sum * dzOuter
            # collect terms
            time_to_hit_zf = 1 / den * (num_A_prefactor * num_A_postfactor - num_B_prefactor * num_B_postfactor)
        else:
            # gardiner p139, eqn 5.2.160
            time_to_hit_zf = 0
            for i, z in enumerate(z_arr[:-1]):
                if z > int_upper:
                    break
                if z > init_z:
                    zmidOuter = (z_arr[i+1] + z_arr[i]) / 2
                    dzOuter = z_arr[i+1] - z_arr[i]
                    factor_B_sum = 0
                    for j, z in enumerate(z_arr[:-1]):
                        if z > zmidOuter:
                            break
                        if z > int_lower:
                            dzInner = z_arr[j + 1] - z_arr[j]
                            #expval_diff = expval_table[j] - expval_table[i]
                            factor_B_sum += np.exp(expval_table[j] - expval_table[i]) / B_table[j] * dzInner
                    time_to_hit_zf += factor_B_sum * dzOuter
        print 'time_to_hit_boundary for Nval', Nval, 'is', time_to_hit_zf
        return time_to_hit_zf

    def prob_to_hit_boundary(Nval, int_lower=0.0, int_upper=1.0, init_z=1.0, hitb=True, fr1=False, fr2=False):
        assert int_lower == 0.0
        #init_z_normed = init_z / Nval
        pmc = params.mod_copy({'N': Nval})

        f_arr, s_arr, z_arr, y_arr = get_centermanifold_traj(pmc, norm=False, force_region_1=fr1, force_region_2=fr2)
        num_pts = len(z_arr)

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
                    intval += A(z_arr[i], y_arr[i], f_arr[i], s_arr[i]) / B(z_arr[i], y_arr[i], f_arr[i], s_arr[i], Nval) * dz
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
                    #intval += 1 / (psi_table[i]) * dz  # TODO this was " 1 / (psi_table[i]) * dz " as of May 19, should not be inverted though
                    intval += (psi_table[i]) * dz
            return intval

        if hitb:
            prob_exit = int_psi(int_lower, init_z) / int_psi(int_lower, int_upper)
        else:
            prob_exit = int_psi(init_z, int_upper) / int_psi(int_lower, int_upper)
        return prob_exit

    if dataid == 'TR1g':
        assert params.mult_inc == 1.0 or params.feedback == 'constant'
        fp_stable = np.array([0, 0, 100.0]) / 100.0
        fp_hidden = np.array([80.76849108597227, 19.262827700935464, -0.03131878690773604]) / 100.0
        yfrac_pt0 = fp_hidden[1]

        init_avg_div = 1.038  # should be (ax + by)/(x+y)566666666666666
        s_renorm = (params.c/init_avg_div) - 1
        print "s_renorm", s_renorm
        pfix = s_renorm

        # this is tau_S heuristic
        # attempts to describe fixation at all-z
        curve_fit_guess = [1 / (params.mu * n * yfrac_pt0 * s_renorm) for n in N_range]
        write_mfpt_heuristic(N_range, curve_fit_guess, filename_mod="_%s_guessPfixTerm1" % dataid)
        curve_fit_guess = [1/(params.mu * n * yfrac_pt0 * s_renorm)         # last factor is 1/pfix
                           + np.log(n * s_renorm) / s_renorm
                           + 0.577/s_renorm     # flux from y->z->zhat
                           for n in N_range]
        write_mfpt_heuristic(N_range, curve_fit_guess, filename_mod="_%s_guessPfixTerm123" % dataid)

        # mfpt to all-z: FPE heuristic
        N_range_timeBoundary = N_range[0:7]  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guessTime1 = [1 / (params.mu * n * yfrac_pt0) + time_to_hit_boundary(n, dual_absorb=False, int_lower=0.0, int_upper=0.5 * n, init_z=0.0)
                                for n in N_range_timeBoundary]
        curve_fit_guessTime2 = [1 / (params.mu * n * yfrac_pt0) + time_to_hit_boundary(n, dual_absorb=False, int_lower=0.0, int_upper=1.0 * n, init_z=0.0)
                                for n in N_range_timeBoundary]
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guessTime1, filename_mod="_%s_guessBoundaryTimeMono1%s" % (dataid, explicit_fname[explicit]))
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guessTime2, filename_mod="_%s_guessBoundaryTimeMono2%s" % (dataid, explicit_fname[explicit]))

        # prob hit boundary heuristic
        N_range_probBoundary = N_range[0:7]
        curve_fit_guessProb2 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=0.5 *  n, init_z=1.0))
                                 for n in N_range_probBoundary]
        curve_fit_guessProb3 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=1.0 *  n, init_z=1.0))
                                 for n in N_range_probBoundary]
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guessProb2, filename_mod="_%s_guessBoundaryProb2%s" % (dataid, explicit_fname[explicit]))
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guessProb3, filename_mod="_%s_guessBoundaryProb3%s" % (dataid, explicit_fname[explicit]))

    elif dataid == 'TR100g':
        assert params.mult_inc == 100.0 and params.feedback != 'constant'
        fp_stable = np.array([41.61623013251053, 41.869771216665875, 16.513998650823595]) / 100.0
        fp_hidden = np.array([71.96914279688974, 28.094892239342407, -0.06403503623214846]) / 100.0

        yfrac_pt0 = fp_hidden[1]
        init_avg_div = 1.056
        #zfrac_pt1 = 0.1643  # solve for x y given gamma such that their mean fitness equals z fitness
        #yfrac_pt1 = 0.4178
        s_max = 0.0854

        s_renorm = (params.c/init_avg_div) - 1
        print "s_renorm", s_renorm

        # this is tau_S heuristic
        # attempts to describe fixation at all-z
        curve_fit_guess = [1 / (params.mu * n * yfrac_pt0 * s_renorm) for n in N_range]
        write_mfpt_heuristic(N_range, curve_fit_guess, filename_mod="_%s_guessPfixTerm1" % dataid)
        curve_fit_guess = [1/(params.mu * n * yfrac_pt0 * s_renorm)         # last factor is 1/pfix
                           + np.log(n * s_renorm) / s_renorm
                           + 0.577/s_renorm     # flux from y->z->zhat
                           for n in N_range]
        write_mfpt_heuristic(N_range, curve_fit_guess, filename_mod="_%s_guessPfixTerm123" % dataid)

        # TRg100 heuristic blobtimes v1  TODO rework
        N_range_dense = np.logspace(np.log10(N_range[0]), np.log10(N_range[-5]), 2*len(N_range))
        blobtimes = [get_blobtime(n,outer_int_upper=None) for n in N_range_dense]
        print "TR100 blobtimes"
        for idx, n in enumerate(N_range_dense):
            print idx, n, blobtimes[idx]
        curve_fit_guess = [1/(params.mu * n * yfrac_pt0 * (1 - np.exp(-params.mu * blobtimes[idx]**2)))  # last factor is pfix
                           for idx, n in enumerate(N_range_dense)]
        write_mfpt_heuristic(N_range_dense, curve_fit_guess, filename_mod="_%s_guessBlobtimes" % dataid)

        # mfpt FPE heuristic
        N_range_timeBoundary = N_range[0:7]  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guessTime1 = [time_to_hit_boundary(n, dual_absorb=False, int_lower=0.0, int_upper=fp_stable[2] * n, init_z=0.0)
                                for n in N_range_timeBoundary]
        print curve_fit_guessTime1
        curve_fit_guessTime2 = [time_to_hit_boundary(n, dual_absorb=False, int_lower=0.0, int_upper=n, init_z=0.0)
                                for n in N_range_timeBoundary]
        print curve_fit_guessTime2
        #curve_fit_guessTime3 = [1 / (params.mu * n * yfrac_pt0 *
        #                         time_to_hit_boundary(n, dual_absorb=True, int_lower=0.0, int_upper=fp_stable[2], init_z=1.0))
        #                         for n in N_range_timeBoundary]
        #curve_fit_guessTime4 = [1 / (params.mu * n * yfrac_pt0 *
        #                         time_to_hit_boundary(n, dual_absorb=True, int_lower=0.0, int_upper=None, init_z=1.0))
        #                         for n in N_range_timeBoundary]
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guessTime1, filename_mod="_%s_guessBoundaryTimeMono1NoDiv%s" % (dataid, explicit_fname[explicit]))
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guessTime2, filename_mod="_%s_guessBoundaryTimeMono2NoDiv%s" % (dataid, explicit_fname[explicit]))
        #write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guessTime3, filename_mod="_%s_guessBoundaryTimeDual1%s" % (dataid, explicit_fname[explicit]))
        #write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guessTime4, filename_mod="_%s_guessBoundaryTimeDual2%s" % (dataid, explicit_fname[explicit]))

        # prob hit boundary heuristic
        print "TR100g prob hit boundary heuristic"
        N_range_probBoundary = N_range[0:7]  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guessProb1 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=fp_stable[2] * n, init_z=1.0))
                                 for n in N_range]
        curve_fit_guessProb2 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=0.5 * n, init_z=1.0))
                                 for n in N_range_probBoundary]
        curve_fit_guessProb3 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=1.0 * n, init_z=1.0))
                                 for n in N_range_probBoundary]
        write_mfpt_heuristic(N_range, curve_fit_guessProb1, filename_mod="_%s_guessBoundaryProb1NoDiv%s" % (dataid, explicit_fname[explicit]))
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guessProb2, filename_mod="_%s_guessBoundaryProb2NoDiv%s" % (dataid, explicit_fname[explicit]))
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guessProb3, filename_mod="_%s_guessBoundaryProb3NoDiv%s" % (dataid, explicit_fname[explicit]))

    elif dataid == 'TR4g':
        assert params.mult_inc == 4.0 and params.feedback != 'constant'
        fp_stable = np.array([24.588047090562984, 24.63656778168078, 50.77538512775624]) / 100.0
        fp_hidden = np.array([80.37960284434968, 19.652759713453463, -0.03236255780313968]) / 100.0

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

        # blobtime heuristic  # TODO make explicit version
        """
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
        """

        # mfpt FPE heuristic from Gardiner
        N_range_timeBoundary = N_range[0:8]  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        #curve_fit_guess1 = [time_to_hit_boundary(n, dual_absorb=False, int_lower=0.0, int_upper=fp_mid[2], init_z=0.0) for n in N_range_timeBoundary]
        curve_fit_guess2 = [time_to_hit_boundary(n, dual_absorb=False, int_lower=0.0, int_upper=n, init_z=0.0) for n in N_range_timeBoundary]
        #curve_fit_guess3 = [1 / (params.mu * n * yfrac_pt0 *
        #                         time_to_hit_boundary(n, dual_absorb=True, int_lower=0.0, int_upper=fp_mid[2], init_z=1.0))
        #                         for n in N_range_timeBoundary]
        #curve_fit_guess4 = [1 / (params.mu * n * yfrac_pt0 *
        #                         time_to_hit_boundary(n, dual_absorb=True, int_lower=0.0, int_upper=None, init_z=1.0))
        #                         for n in N_range_timeBoundary]
        #write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guess1, filename_mod="_%s_guessBoundaryTimeMono1%s" % (dataid, explicit_fname[explicit]))
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guess2, filename_mod="_%s_guessBoundaryTimeMono2%s" % (dataid, explicit_fname[explicit]))
        #write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guess3, filename_mod="_%s_guessBoundaryTimeDual1%s" % (dataid, explicit_fname[explicit]))
        #write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guess4, filename_mod="_%s_guessBoundaryTimeDual2%s" % (dataid, explicit_fname[explicit]))

        # prob hit boundary heuristic
        N_range_probBoundary = N_range[0:7]  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        #curve_fit_guess1 = [1 / (params.mu * n * yfrac_pt0 * prob_to_hit_boundary(n, int_lower=0.0, int_upper=fp_mid[2], init_z=1.0))
        #                         for n in N_range_probBoundary]
        #curve_fit_guess2 = [1 / (params.mu * n * yfrac_pt0 * prob_to_hit_boundary(n, int_lower=0.0, int_upper=0.4, init_z=1.0))
        #                         for n in N_range_probBoundary]
        curve_fit_guess3 = [1 / (params.mu * n * yfrac_pt0 * prob_to_hit_boundary(n, int_lower=0.0, int_upper=1.0 * n, init_z=1.0))
                                 for n in N_range_probBoundary]
        #write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess1, filename_mod="_%s_guessBoundaryProb1%s" % (dataid, explicit_fname[explicit]))
        #write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess2, filename_mod="_%s_guessBoundaryProb2%s" % (dataid, explicit_fname[explicit]))
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess3, filename_mod="_%s_guessBoundaryProb3%s" % (dataid, explicit_fname[explicit]))

    elif dataid == 'BL4g':
        assert params.mult_inc == 4.0 and params.feedback != 'constant'
        fp_low = np.array([85.08181731871301, 14.89695736662967, 0.02122531465732358]) / 100.0
        fp_mid = np.array([21.56844087406341, 21.53060213939143, 56.900956986545154]) / 100.0
        yfrac_pt0 = fp_low[1]

    elif dataid == 'BL1g':
        print params.mult_inc, params.mult_dec
        assert params.mult_inc == 1.0 or params.feedback == 'constant'
        fp_low = np.array([85.39353129821689, 14.585869420527702, 0.02059928125541255]) / 100.0
        fp_mid = np.array([0,0,100.0]) / 100.0
        yfrac_pt0 = fp_low[1]

        # mfpt FPE heuristic from Gardiner
        N_range_timeBoundary = N_range[0:8]
        curve_fit_guess2 = [time_to_hit_boundary(n, dual_absorb=False, int_lower=0.0, int_upper=n, init_z=0.0) for n in N_range_timeBoundary]
        write_mfpt_heuristic(N_range_timeBoundary, curve_fit_guess2, filename_mod="_%s_guessBoundaryTimeMono2%srev" % (dataid, explicit_fname[explicit]))

        # prob hit boundary heuristic (not top, index 3, is the only barrier point to consider here)
        N_range_probBoundary = N_range[0:7]  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guess3 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=1.0 * n, init_z=1.0))
                                 for n in N_range_probBoundary]
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess3, filename_mod="_%s_guessBoundaryProb3%srev" % (dataid, explicit_fname[explicit]))

    else:
        assert dataid not in fnames.keys()
        print "Warning: %s dataid not in fnames.keys()" % dataid

        assert params.mult_inc in [1.0, 100.0]
        assert dataid in ['1region1g', '1region100g', '2region1g', '2region100g']

        fp_lows = {
            '1region1g': np.array([83.32060255912937, 16.662731931547082, 0.016665509323548378]) / 100.0,
            '1region100g': np.array([74.88478244041383, 25.090121140143346, 0.025096419442824924]) / 100.0,
            '2region1g': np.array([83.34837879374416, 16.66828671556507, -0.01666550930923094]) / 100.0,
            '2region100g': np.array([74.96028867279259, 25.064769817738842, -0.02505849053142839]) / 100.0}
        force_region_flags = {'1region1g': (True, False),
                              '1region100g': (True, False),
                              '2region1g': (False, True),
                              '2region100g': (False, True)}
        fp_low = fp_lows[dataid]
        fr1, fr2 = force_region_flags[dataid]

        yfrac_pt0 = fp_low[1]
        # MASTER EQN SOLVE BLOCK
        N_range_alt = N_range[0:9]
        curve_fit_linalg1 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=False, force_region_1=fr1,
                                         force_region_2=fr2, y_0_frac_override=yfrac_pt0)[0] for n in N_range_alt]
        curve_fit_linalg2 = [linalg_mfpt(params=params.mod_copy({'N': n}), flag_zhat=True, force_region_1=fr1,
                                         force_region_2=fr2, y_0_frac_override=yfrac_pt0)[0] for n in N_range_alt]
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg1, filename_mod="_%s_linalgALLZ" % dataid)
        write_mfpt_heuristic(N_range_alt, curve_fit_linalg2, filename_mod="_%s_linalgZHAT" % dataid)

        # prob hit boundary heuristic
        N_range_probBoundary = N_range[0:8]  # np.logspace(np.log10(N_range[0]), np.log10(N_range[-1]), 1*len(N_range))
        curve_fit_guess1 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=0.2, init_z=1.0, fr1=fr1, fr2=fr2))
                                 for n in N_range_probBoundary]
        curve_fit_guess2 = [1 / (params.mu * n * yfrac_pt0 *
                                 prob_to_hit_boundary(n, int_lower=0.0, int_upper=1.0, init_z=1.0, fr1=fr1, fr2=fr2))
                                 for n in N_range_probBoundary]
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess1, filename_mod="_%s_guessBoundaryProb1" % dataid)
        write_mfpt_heuristic(N_range_probBoundary, curve_fit_guess2, filename_mod="_%s_guessBoundaryProb2" % dataid)

    return


if __name__ == '__main__':
    param_varying_name = "N"
    assert param_varying_name == "N"

    # pick form of transition rates
    EXPLICIT = True
    SKIP_LINALG = True
    ALT_ABSORB = True

    # pick one of the four presets:
    data_id = 'BL1g'
    params = presets(data_id)  # TODO generalize preset in main args

    # N_range = [int(a) for a in np.logspace(1.50515, 4.13159, num=11)]  # <-- current default
    N_range = [int(a) for a in np.logspace(1.50515, 4.13159, num=11)] #+ [int(a) for a in np.logspace(4.8, 7, num=4)]
    print "Running trajectory analysis for %s" % data_id
    print "N_range %d points, bounds:" % len(N_range), np.min(N_range), np.max(N_range)

    generate_heuristic_mfpt(params, N_range, 'N', data_id, explicit=EXPLICIT, skip_linalg=SKIP_LINALG, alt_absorb=ALT_ABSORB)
