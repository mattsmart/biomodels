"""
Conventions
- params is 7-vector of the form: params[0] -> alpha_plus
                                  params[1] -> alpha_minus
                                  params[2] -> mu
                                  params[3] -> a           (usually normalized to 1)
                                  params[4] -> b           (b = 1 - delta)
                                  params[5] -> c           (c = 1 + s)
                                  params[6] -> N           (float not int)
                                  params[7] -> v_x
                                  params[8] -> v_y         (typically 0)
                                  params[9] -> v_z         (typically 0)
                                  params[10] -> mu_base    (typically 0)
- if an element of params is specified as None then a bifurcation range will be be found and used
"""

import numpy as np
from random import random
from scipy.integrate import ode, odeint
from scipy.optimize import approx_fprime, fsolve
from sympy import Symbol, solve, re

from constants import PARAMS_ID, CSV_DATA_TYPES, SIM_METHODS_VALID, INIT_COND, TIME_START, TIME_END, NUM_STEPS
from params import Params


def ode_system_vector(init_cond, times, params):
    dxvec_dt = params.ode_system_vector(init_cond, times)
    return dxvec_dt


def system_vector_obj_ode(t_scalar, r_idx, params):
    return ode_system_vector(r_idx, t_scalar, params)


def ode_euler(init_cond, times, params):
    dt = times[1] - times[0]
    r = np.zeros((len(times), params.numstates))
    r[0] = np.array(init_cond)
    for idx, t in enumerate(times[:-1]):
        v = ode_system_vector(r[idx], None, params)
        r[idx+1] = r[idx] + np.array(v)*dt
    return r, times


def ode_rk4(init_cond, times, params):
    dt = times[1] - times[0]
    r = np.zeros((len(times), params.numstates))
    r[0] = np.array(init_cond)
    obj_ode = ode(system_vector_obj_ode, jac=None)
    obj_ode.set_initial_value(init_cond, times[0])
    obj_ode.set_f_params(params)
    obj_ode.set_integrator('dopri5')
    idx = 1
    while obj_ode.successful() and obj_ode.t < times[-1]:
        obj_ode.integrate(obj_ode.t + dt)
        r[idx] = np.array(obj_ode.y)
        idx += 1
    return r, times


def ode_libcall(init_cond, times, params):
    fn = ode_system_vector
    r = odeint(fn, init_cond, times, args=(params,))
    return r, times


def reaction_propensities(r, step, params, fpt_flag=False):
    rxn_prop = params.rxn_prop(r[step])
    if fpt_flag:
        laststate_idx = params.numstates - 1
        laststate = r[step][laststate_idx]
        rxn_prop.append(params.mu*laststate)  # special transition events for z1->z2 (extra mutation)
    return rxn_prop


def reaction_propensities_lowmem(current_state, params, fpt_flag=False):
    rxn_prop = params.rxn_prop(current_state)
    if fpt_flag:
        laststate_idx = params.numstates - 1
        laststate = current_state[laststate_idx]
        rxn_prop.append(params.mu*laststate)  # special transition events for z1->z2 (extra mutation)
    return rxn_prop


def bisecting_rxn_search_iter(propensities, L, R, T, m=0):
    while L<=R:
        m = int(np.floor((L + R) / 2))
        if propensities[m] <= T:
            L=m+1
        else:
            R=m-1
    return m


def bisecting_rxn_search_recurse(propensities, L, R, T, m=0):
    if L > R:
        return m
    m = int(np.floor((L + R) / 2))
    if propensities[m] <= T:
        return bisecting_rxn_search_recurse(propensities, m+1, R, T, m=m)
    else:
        return bisecting_rxn_search_recurse(propensities, L, m-1, T, m=m)


def stoch_gillespie(init_cond, num_steps, params, fpt_flag=False, establish_flag=False):
    """
    There are 12 transitions to consider:
    - 6 birth/death of the form x_n -> x_n+1, (x birth, x death, ...), label these 0 to 5
    - 3 transitions of the form x_n -> x_n-1, (x->y, y->x, y->z), label these 6 to 8
    - 3 transitions associated with immigration (vx, vy, vz), label these 9 to 11
    - 1 transitions for x->z (rare), label this 12
    Gillespie algorithm has indefinite timestep size so consider total step count as input (times input not used)
    Notes on fpt_flag:
    - if fpt_flag (first passage time) adds extra rxn propensity for transition from z1->z2
    - return r[until fpt], times[until fpt]
    Notes on establish_flag:
    - can't be on same time as fpt flag
    - in x,y,z model, return time when first successful z was generated (success = establish)
    """
    assert not (fpt_flag and establish_flag)
    time = 0.0
    r = np.zeros((num_steps, params.numstates))
    times_stoch = np.zeros(num_steps)
    r[0, :] = np.array(init_cond, dtype=int)  # note stochastic sim operates on integer population counts
    update_dict = params.update_dict
    fpt_rxn_idx = len(update_dict.keys()) - 1  # always use last element as special FPT event
    fpt_event = False
    establish_event = False
    for step in xrange(num_steps-1):
        r1 = random()  # used to determine time of next reaction
        r2 = random()  # used to partition the probabilities of each reaction
        # compute propensity functions (alpha) and the partitions for all 12 transitions
        alpha = reaction_propensities(r, step, params, fpt_flag=fpt_flag)
        alpha_partitions = np.zeros(len(alpha)+1)
        alpha_sum = 0.0
        for i in xrange(len(alpha)):
            alpha_sum += alpha[i]
            alpha_partitions[i + 1] = alpha_sum
        #alpha_partitions = alpha_partitions / alpha_sum  #rescale r2 instead to save cycles

        # find time to first reaction
        tau = np.log(1 / r1) / alpha_sum  # remove divison to neg? faster?

        # BISECTING SEARCH METHOD (slower for small number of reactions)
        #r2_scaled = alpha_sum * r2
        #rxn_idx = bisecting_rxn_search(alpha_partitions, 0, len(alpha_partitions), r2_scaled)
        #pop_updates = update_dict[rxn_idx]
        #r[step + 1] = r[step] + pop_updates

        #DIRECT SEARCH METHOD (faster for 14 or fewer rxns so far)
        r2_scaled = alpha_sum*r2
        for rxn_idx in xrange(len(alpha)):
            if alpha_partitions[rxn_idx] <= r2_scaled < alpha_partitions[rxn_idx + 1]:  # i.e. rxn_idx has occurred
                pop_updates = update_dict[rxn_idx]
                r[step+1] = r[step] + pop_updates
                break

        time += tau
        times_stoch[step + 1] = time

        """
        if step % 10000 == 0:
            print "step", step, "time", time, ":", r[step,:], "to", r[step+1, :]
            if step >= 100000:
                return None, None
        """

        if establish_flag and (r[step + 1][-1] >= params.N):
            establish_event = True

        if rxn_idx == fpt_rxn_idx:
            assert fpt_flag                             # just in case, not much cost
            return r[:step+2, :], times_stoch[:step+2]  # end sim because fpt achieved

        if establish_event:
            assert establish_flag                       # just in case, not much cost
            return r[:step+2, :], times_stoch[:step+2]  # end sim because fpt achieved

    if fpt_flag or establish_flag:  # if code gets here should recursively continue the simulation
        init_cond = r[-1]
        r_redo, times_stoch_redo = stoch_gillespie(init_cond, num_steps, params, fpt_flag=fpt_flag, establish_flag=establish_flag)
        times_stoch_redo_shifted = times_stoch_redo + times_stoch[-1]  # shift start time of new sim by last time
        return np.concatenate((r, r_redo)), np.concatenate((times_stoch, times_stoch_redo_shifted))
    return r, times_stoch


def stoch_gillespie_lowmem(init_cond, params, init_time=0, fpt_flag=False, establish_flag=False):
    assert not (fpt_flag and establish_flag)

    update_dict = params.update_dict
    fpt_rxn_idx = len(update_dict.keys()) - 1  # always use last element as special FPT event

    current_state = np.array(init_cond)
    current_time = init_time
    exit_flag = False

    while not exit_flag:
        r1 = random()  # used to determine time of next reaction
        r2 = random()  # used to partition the probabilities of each reaction

        # compute propensity functions (alpha) and the partitions for all 12 transitions
        alpha = reaction_propensities_lowmem(current_state, params, fpt_flag=fpt_flag)
        alpha_partitions = np.zeros(len(alpha) + 1)
        alpha_sum = 0.0
        for i in xrange(len(alpha)):
            alpha_sum += alpha[i]
            alpha_partitions[i + 1] = alpha_sum

        # find time to first reaction
        tau = np.log(1 / r1) / alpha_sum  # remove divison to neg? faster?
        current_time += tau

        # DIRECT SEARCH METHOD (faster for 14 or fewer rxns so far)
        r2_scaled = alpha_sum * r2
        for rxn_idx in xrange(len(alpha)):
            if alpha_partitions[rxn_idx] <= r2_scaled < alpha_partitions[rxn_idx + 1]:  # i.e. rxn_idx has occurred
                pop_updates = update_dict[rxn_idx]
                current_state += pop_updates
                break

        # fpt exit conditions
        if fpt_flag:
            if fpt_rxn_idx == rxn_idx:
                current_state[2] += 1  # Note: this is so that final pop hasn't been decremented from the z->z2 swap
                exit_flag = True

        # establish exit condition
        if establish_flag:
            if current_state[-1] >= params.N:
                exit_flag = True

    return current_state, current_time


def stoch_bnb(init_cond, num_steps, params, fpt_flag=False, establish_flag=False):
    # uses modified version of BNB algorithm for evolutionary dynamics (2012)
    # - exact algo assumes constant linear birth and death rates, here our death rates are functions of the state
    # - approx by using smaller timestep and updating within
    # TODO check optimize: less frequent updates in Supp info of paper
    # TODO fix issue where x rapidly goes extinct, or auto extinct usually on first timestep...

    assert fpt_flag == False  # TODO not implmented, but test once done
    assert params.numstates == 3  # TODO generalize check
    assert params.mu_base == 0    # need to modify transition tables to include for xyz case otherwise

    def p_M_of_t(time_var, rate_birth, rate_death, rate_trans):
        R = np.sqrt((rate_birth - rate_death) ** 2 + rate_trans * (2 * rate_birth + 2 * rate_death + rate_trans))
        W = rate_birth + rate_death + rate_trans
        C = np.cosh(R*time_var/2)  # see SI for 2012 paper
        S = np.sinh(R*time_var/2)
        num = R*C + 2*rate_death*S - W*S
        den = R*C - 2*rate_birth*S + W*S
        return num/den

    def p_E_of_t(time_var, rate_birth, rate_death, rate_trans, p_M_of_t_val):
        W = rate_birth + rate_death + rate_trans
        num = rate_death * (1 - p_M_of_t_val)
        den = W - rate_death - rate_birth * p_M_of_t_val
        return num/den

    def p_B_of_t(time_var, rate_birth, rate_death, rate_trans, p_E_of_t_val):
        return rate_birth * p_E_of_t_val / rate_death


    def eqn_11(n_class, rate_birth, rate_death, rate_trans):
        if n_class == 0.0:
            t_next_class_swap = 1e9  # very big bc not possible
        else:
            r = random()
            R = np.sqrt((rate_birth - rate_death)**2 + rate_trans * (2*rate_birth + 2*rate_death + rate_trans))
            W = rate_birth + rate_death + rate_trans
            print "eqn11: r, R, W, rate_birth, rate_death, rate_trans"
            print r, R, W, rate_birth, rate_death, rate_trans
            eqn_12_expr = ((R - W + 2*rate_death) / (R + W - 2*rate_birth)) ** n_class
            if eqn_12_expr > r:
                print "WARNING\neqn_12_expr > r ||", eqn_12_expr, ">", r,"\nWARNING"
                t_next_class_swap = 1e9
            else:
                r_scaled = r ** (1/n_class)
                ln_num = r_scaled * (R - W + 2*rate_birth) - W - R + 2 * rate_death
                ln_den = r_scaled * (-R - W + 2*rate_birth) - W + R + 2 * rate_death
                print r, R, W, r_scaled, ln_num, ln_den, np.log(ln_num/ln_den)
                t_next_class_swap = 1/R * np.log(ln_num/ln_den)
            r_scaled = r ** (1 / n_class)
            ln_num = r_scaled * (R - W + 2 * rate_birth) - W - R + 2 * rate_death
            ln_den = r_scaled * (-R - W + 2 * rate_birth) - W + R + 2 * rate_death
            print "eqn11: r, R, W, r_scaled, ln_num, ln_den, np.log(ln_num / ln_den)"
            print r, R, W, r_scaled, ln_num, ln_den, np.log(ln_num / ln_den)
            t_next_class_swap = 1 / R * np.log(ln_num / ln_den)

        return t_next_class_swap

    def update_trans_rates_from_state(params, trans_rates, state):
        mod_dict = params.system_variants(state, None)
        for k,v in mod_dict.iteritems():
            for idx in params.transrates_param_to_key[k]:
                trans_rates[idx] = v
        return trans_rates

    def trans_rates_to_per_class(params, trans_rates):
        trans_rates_per_class = np.zeros(params.numstates)
        for k in xrange(params.numstates):
            class_alloutparams = params.transrates_class_to_alloutparams[k]
            for param_name in class_alloutparams:
                param_idx_as_list = params.transrates_param_to_key[param_name]
                trans_rates_per_class[k] += trans_rates[param_idx_as_list[0]]  # all idx in list correspond to same param val
        return trans_rates_per_class

    def choose_transition_event(params, trans_rates, trans_rates_per_class, class_idx):
        map_class_to_rxn = params.transrates_class_to_rxnidx
        if len(map_class_to_rxn[class_idx]) == 1:
            event_idx = map_class_to_rxn[class_idx][0]
        else:
            # gillespie-like procedure here
            associated_rxn_idx = params.transrates_class_to_rxnidx
            r = random()
            r_scaled = r * trans_rates_per_class[class_idx]  # TODO assert is sum of [trans rates associated_rxn_idx's]
            start = 0
            for rxn_idx in associated_rxn_idx:
                next = start + trans_rates[rxn_idx]
                print "start < r_scaled < next:",
                print start, r_scaled, next
                if start < r_scaled < next:
                    event_idx = rxn_idx
                    break
                else:
                    start = next
        return event_idx

    # build rate info
    birth_rates = params.growthrates[:]
    trans_rates = params.transrates_base[:]
    times_each_transition = np.zeros(params.numstates)

    # time and state vectors for storage
    time = 0.0
    state = np.zeros((num_steps, params.numstates))
    times_stoch = np.zeros(num_steps)
    state[0, :] = np.array(init_cond, dtype=int)  # note stochastic sim operates on integer population counts

    for step in xrange(num_steps-1):  # each step find time to next mutation (here means class transition)

        current_state = state[step, :]
        # update transition propensities based on current state
        trans_rates = update_trans_rates_from_state(params, trans_rates, current_state)
        trans_rates_per_class = trans_rates_to_per_class(params, trans_rates)
        fbar = params.fbar(current_state)
        for k in xrange(params.numstates):
            times_each_transition[k] = eqn_11(current_state[k], birth_rates[k], fbar, trans_rates_per_class[k])
        print "\n\ncurrent_state", current_state
        print "trans_rates", trans_rates
        print "trans_rates_per_class, fbar", trans_rates_per_class, fbar
        print "times_each_transition", times_each_transition

        # identify next transition class
        class_idx = np.argmin(times_each_transition)
        time_to_next_transition = times_each_transition[class_idx]
        print "time_to_next_transition", time_to_next_transition
        # update time
        time += time_to_next_transition
        times_stoch[step+1] = time

        # binomial negative-binomial part
        bin_n = current_state[class_idx] - 1
        p_M_at_t = p_M_of_t(time_to_next_transition, birth_rates[class_idx], fbar, trans_rates_per_class[class_idx])
        p_E_at_t = p_E_of_t(time_to_next_transition, birth_rates[class_idx], fbar, trans_rates_per_class[class_idx], p_M_at_t)
        bin_p = 1 - p_E_at_t / p_M_at_t
        print "p_M_at_t, p_E_at_t, bin_p", p_M_at_t, p_E_at_t, bin_p
        m_binomial = np.random.binomial(bin_n, bin_p)
        print "for mutating class %d have m %.2f" % (class_idx, m_binomial)

        negbin_n = m_binomial + 2
        negbin_p = p_B_of_t(time_to_next_transition, birth_rates[class_idx], fbar, trans_rates_per_class[class_idx], p_E_at_t)
        updated_class_amount = 1 + m_binomial + np.random.negative_binomial(negbin_n, negbin_p)

        state[step+1, class_idx] = updated_class_amount

        # their step 5 update all other pops according to algo 2
        for k in xrange(params.numstates):
            if k != class_idx:
                bin_n = current_state[k]
                if bin_n == 0:
                    updated_class_amount = 0
                else:
                    rate_trans = trans_rates_per_class[k]
                    p_M_at_t = p_M_of_t(time_to_next_transition, birth_rates[k], fbar, rate_trans)
                    p_E_at_t = p_E_of_t(time_to_next_transition, birth_rates[k], fbar, rate_trans, p_M_at_t)
                    bin_p = 1 - p_E_at_t / p_M_at_t
                    m_binomial = np.random.binomial(bin_n, bin_p)
                    if m_binomial == 0:
                        print "If m_binomial=0, then the system at time t is in the extinct state n_class_updated=0"
                        updated_class_amount = 0
                    else:
                        negbin_n = m_binomial
                        negbin_p = p_B_of_t(time_to_next_transition, birth_rates[class_idx], fbar, rate_trans, p_E_at_t)
                        updated_class_amount = m_binomial + np.random.negative_binomial(negbin_n, negbin_p)
                state[step + 1, k] = updated_class_amount

        # increment class where transition went to\
        # TODO: need to pick which reaction occurs from class_allout_rates and update accordingly here somehow
        trans_event_num = choose_transition_event(params, trans_rates, trans_rates_per_class, class_idx)
        label, class_idx, class_idx_after = params.transition_dict[trans_event_num]
        state[step + 1, class_idx_after] += 1

        if establish_flag:
            if state[step + 1, -1] >= params.N:
                return state[:step + 2, :], times_stoch[:step + 2]  # end sim because est achieved

    if fpt_flag or establish_flag:  # if code gets here should recursively continue the simulation
        init_cond = state[-1, :]
        state_redo, times_stoch_redo = stoch_bnb(init_cond, num_steps, params, fpt_flag=fpt_flag, establish_flag=establish_flag)
        times_stoch_redo_shifted = times_stoch_redo + times_stoch[-1]  # shift start time of new sim by last time
        return np.concatenate((state, state_redo)), np.concatenate((times_stoch, times_stoch_redo_shifted))

    return state, times_stoch


def stoch_tauleap_adaptive(init_cond, num_steps, params, fpt_flag=False, establish_flag=False):
    assert 1 == 0  # broken
    # TODO NOTE also speedup exact SSA FPT by not tracking whole array just last and next of time and state
    # TODO can also approx tauleap in BNB algo use dt=1e-2 as used in fisher paper
    assert not (fpt_flag and establish_flag)
    inv_propensity_multiple = 10   # default: 10
    num_ssa_steps = 100           # default: 100

    # choose tau (eps, nc params from original 2006 paper)
    tau_override = 1e-2  # used in 2009 fisher
    flag_tau_override = True
    if flag_tau_override:
        flag_normalize_each_step = True
        print "tau_override with tau %.2f" % tau_override
    else:
        flag_normalize_each_step = False

    def normalize_state(current_state, params):
        scale = params.N / sum(current_state)
        state_normalized = [int(current_state[k] * scale + 0.5) if current_state[k] >= 0 else 0
                            for k in xrange(params.numstates)]
        """
        #print "quickly normalizing popsize...", current_state, "to", state_normalized
        if current_state[-1] > 0 and state_normalized[-1] == 0:
            print "\nWARNING state[k+1, -1]=%d > 0 and state_normalized[-1] == 0 :\n" % current_state[-1]
            print "quickly normalizing popsize...", current_state, "to", state_normalized
            assert 1 == 0
        """
        return state_normalized

    def calc_rxn_events(delta_t, propensities, update_vec_array_transpose):
        poisson_params = propensities * delta_t
        amount_each_rxn_in_step = np.random.poisson(poisson_params)
        state_increment = np.dot(update_vec_array_transpose, amount_each_rxn_in_step)
        return state_increment, amount_each_rxn_in_step

    def choose_tau(current_state, params, propensities, num_crit=10, eps=0.1):
        # identify critical and noncrit rxns
        crit_rxns = []
        for rxn_idx in xrange(len(propensities)):
            alpha_j = propensities[rxn_idx]
            decrement_idx = np.argmin(params.update_dict[rxn_idx])  # TODO SPEEDUP WARNING assumes only one cell can be decremented in one reaction (check params.update_dict)
            L_j = current_state[decrement_idx]
            exhaust_cond = L_j <= num_crit  # exhaustion condition
            if (alpha_j > 0) and exhaust_cond:
                crit_rxns.append(rxn_idx)
        non_crit_rxns = [rxn_idx for rxn_idx in xrange(len(propensities)) if rxn_idx not in crit_rxns]

        # candidate tau selection step
        tau_candidates = np.zeros(params.numstates)
        for k in xrange(params.numstates):
            n_k = current_state[k]
            g_k = 1  # TODO currently assumes all first order rxns is this true? death rate sorta quadratic but treat as updating linear?
            mu_k = 0
            sigma_k = 0
            pair = [0, 0]
            for rxn_idx in non_crit_rxns:
                update_vector_j = params.update_dict[rxn_idx]
                mu_k += update_vector_j[k] * propensities[rxn_idx]
                sigma_k += (update_vector_j[k] ** 2) * propensities[rxn_idx]
            tau_numerator = np.max([1, eps * n_k / g_k])
            pair[0] = tau_numerator / np.abs(mu_k)
            pair[1] = (tau_numerator / sigma_k) ** 2
            tau_candidates[k] = np.min(pair)
        print "tau_candidates:", tau_candidates
        tau_candidate = np.min(tau_candidates)

        return tau_candidate

    time = 0.0
    state = np.zeros((num_steps, params.numstates))
    times_stoch = np.zeros(num_steps)
    state[0, :] = np.array(init_cond, dtype=int)  # note stochastic sim operates on integer population counts

    fpt_rxn_idx = len(params.update_dict.keys()) - 1  # always use last element as special FPT event
    fpt_event = False
    establish_event = False

    num_rxn = len(params.update_dict.keys()) - 1
    if fpt_flag:
        num_rxn = num_rxn + 1
    update_vec_array = np.array([params.update_dict[key] for key in xrange(num_rxn)])
    update_vec_array_transpose = np.transpose(update_vec_array)

    for step in xrange(num_steps - 1):
        current_state = state[step,:]

        # compute propensity functions (alpha) and the partitions for all 12 transitions
        alpha = np.array(reaction_propensities(state, step, params, fpt_flag=fpt_flag))
        alpha_sum = np.sum(alpha)
        tau_cutoff = inv_propensity_multiple / alpha_sum

        # choose tau
        if flag_tau_override:
            tau_main_candidate = tau_override
        else:
            tau_main_candidate = choose_tau(current_state, params, alpha)  # TODO choose or adaptive fn

        # if tau too small, do some exact steps
        # exit criterion

        if (not flag_tau_override) and tau_main_candidate < tau_cutoff:  # do some exact steps and restart loop
            print "WARNING: NOT tau_override, and tau candidate %.5f < tau_cutoff %.5f" % (tau_main_candidate, tau_cutoff)
            substate, subtimes = stoch_gillespie(current_state, num_ssa_steps, params, fpt_flag=False, establish_flag=False)
            # update tracking arrays
            state[step + 1, :] = substate[-1, :]
            time += subtimes[-1]
            times_stoch[step + 1] = time
        else:
            # TODO implement tau = min tau1 tau2 from second part of algo
            """
            scale = 1/alpha_sum
            tau_alt_candidate = np.random.exponential(scale)
            if tau_main_candidate < tau_alt_candidate:
                # tau = tau_main_candidate
                # only update non crit rxn events num, crit are all 0
            else:
                tau = tau_alt_candidate
                # longer paragraph....
            """

            # compute poissonian event counts for all j rxn with param lambda = aj(t) * tau
            state_increment, amount_each_rxn_in_step = calc_rxn_events(tau_main_candidate, alpha, update_vec_array_transpose)
            if fpt_flag:
                if amount_each_rxn_in_step[fpt_rxn_idx] > 0:
                    fpt_event = True
            #print "amount_each_rxn_in_step"
            #print amount_each_rxn_in_step

            # update tracking arrays
            state[step+1, :] = state[step,:] + state_increment
            if flag_normalize_each_step:
                state[step + 1, :] = normalize_state(state[step+1, :], params)
            time += tau_main_candidate
            times_stoch[step + 1] = time

        if step % 10000 == 0:
            print "step", step, "time", time, ":", state[step,:], "to", state[step+1, :]
            if step >= 500000:
                return None, None

        # fpt and establish exit conditions
        if fpt_event:
            assert fpt_flag                                 # just in case, not much cost
            return state[:step+2, :], times_stoch[:step+2]  # end sim because fpt achieved

        if establish_flag and state[step+1, -1] >= params.N:
            return state[:step+2, :], times_stoch[:step+2]  # end sim because fpt achieved

    if fpt_flag or establish_flag:  # if code gets here should recursively continue the simulation
        #print "recursing in tauleap to wait for event flag exit condition"
        init_cond = state[-1, :]
        state_redo, times_stoch_redo = stoch_tauleap(init_cond, num_steps, params, fpt_flag=fpt_flag, establish_flag=establish_flag)
        times_stoch_redo_shifted = times_stoch_redo + times_stoch[-1]  # shift start time of new sim by last time
        return np.concatenate((state, state_redo)), np.concatenate((times_stoch, times_stoch_redo_shifted))

    return state, times_stoch


def stoch_tauleap(init_cond, num_steps, params, fpt_flag=False, establish_flag=False, recurse=0):
    # TODO NOTE also speedup exact SSA FPT by not tracking whole array just last and next of time and state
    # TODO can also approx tauleap in BNB algo use dt=1e-2 as used in fisher paper
    # TODO make 2 diff tau leap fn bc BIG slowdown when doing this brief combo thing, gillespie actually faster
    assert not (fpt_flag and establish_flag)

    # choose tau (eps, nc params from original 2006 paper)
    tau_override = 1e-2  # used in 2009 fisher
    flag_normalize_each_step = True

    def calc_rxn_events(state, step, delta_t, propensities, update_vec_array_transpose):
        """  OLD, looks nice but slower ~ 30%
        poisson_params = propensities * delta_t
        amount_each_rxn_in_step = np.random.poisson(poisson_params)
        state_increment = np.dot(update_vec_array_transpose, amount_each_rxn_in_step)
        """
        fpt_event = False
        state[step + 1, :] = state[step, :]
        for rxn_idx in xrange(len(propensities)):
            if propensities[rxn_idx] > 0.0:
                poisson_param = propensities[rxn_idx] * delta_t
                amount_rxn = np.random.poisson(poisson_param)
                if rxn_idx == fpt_rxn_idx and amount_rxn > 0:
                    fpt_event = True
                state[step + 1, :] += update_vec_array_transpose[:, rxn_idx] * amount_rxn
        return fpt_event

    state = np.zeros((num_steps, params.numstates))
    times_stoch = np.arange(num_steps) * tau_override
    state[0, :] = np.array(init_cond, dtype=int)  # note stochastic sim operates on integer population counts

    num_rxn = len(params.update_dict.keys()) - 1
    if fpt_flag:
        num_rxn = num_rxn + 1
    update_vec_array = np.array([params.update_dict[key] for key in xrange(num_rxn)])
    update_vec_array_transpose = np.transpose(update_vec_array)
    fpt_rxn_idx = len(params.update_dict.keys()) - 1  # always use last element as special FPT event
    fpt_event = False
    establish_event = False

    for step in xrange(num_steps - 1):
        # compute propensity functions (alpha) and the partitions for all 12 transitions
        alpha = reaction_propensities(state, step, params, fpt_flag=fpt_flag)

        # compute poissonian event counts for all j rxn with param lambda = aj(t) * tau
        fpt_event = calc_rxn_events(state, step, tau_override, alpha, update_vec_array_transpose)
        #print "OUT", state[step, :], state[step+1, :]

        # update tracking arrays
        if flag_normalize_each_step:
            state[step + 1, :] = [int(state[step+1, k] + 0.5) if state[step+1, k] >= 0 else 0
                                  for k in xrange(params.numstates)]

        # temp exit and printing
        #if step % 100000 == 0:
        #    print "step", step, "time", times_stoch[step], ":", state[step,:], "to", state[step+1, :]

        # fpt and establish exit conditions
        if fpt_event:
            assert fpt_flag                                 # just in case, not much cost
            return state[:step+2, :], times_stoch[:step+2]  # end sim because fpt achieved
        if establish_flag and state[step+1, -1] >= params.N:
            return state[:step+2, :], times_stoch[:step+2]

    if fpt_flag or establish_flag:  # if code gets here should recursively continue the simulation
        #print "recursing (%.2f) in tauleap to wait for event flag exit condition" % recurse
        recurse += times_stoch[-1]
        init_cond = state[-1, :]
        state_redo, times_stoch_redo = stoch_tauleap(init_cond, num_steps, params, fpt_flag=fpt_flag,
                                                     establish_flag=establish_flag, recurse=recurse, brief=brief)
        times_stoch_redo_shifted = times_stoch_redo + times_stoch[-1]  # shift start time of new sim by last time
        return np.concatenate((state, state_redo)), np.concatenate((times_stoch, times_stoch_redo_shifted))

    return state, times_stoch


def stoch_tauleap_lowmem(init_cond, num_steps, params, fpt_flag=False, establish_flag=False, init_time=0):
    # TODO make 2 diff tau leap fn bc BIG slowdown when doing this brief combo thing, gillespie actually faster

    assert not (fpt_flag and establish_flag)
    assert len(init_cond) == params.numstates

    # choose tau and normalization settings
    tau_override = 1e-2  # used in 2009 fisher
    flag_normalize_each_step = True

    def calc_rxn_events(current_state, delta_t, propensities, update_vec_array_transpose):
        # TODO annoying optimization is to rewrite rxn prop inside this fn to reduce looping
        fpt_event = False
        for rxn_idx in xrange(len(propensities)):  # TODO check optimization of preprocess prop into (nonzero prop, rxn idx) pairs
            if propensities[rxn_idx] > 0.0:
                poisson_param = propensities[rxn_idx] * delta_t  # only non-zero alpha_j rxn can have events
                amount_rxn = np.random.poisson(poisson_param)
                if rxn_idx == fpt_rxn_idx and amount_rxn > 0:
                    fpt_event = True
                current_state += update_vec_array_transpose[:, rxn_idx] * amount_rxn  # TODO how to optimize?
        return fpt_event

    current_time = init_time
    current_state = np.array([v for v in init_cond])

    num_rxn = len(params.update_dict.keys()) - 1
    if fpt_flag:
        num_rxn = num_rxn + 1
    update_vec_array = np.array([params.update_dict[key] for key in xrange(num_rxn)])
    update_vec_array_transpose = np.transpose(update_vec_array)

    fpt_rxn_idx = len(params.update_dict.keys()) - 1  # always use last element as special FPT event
    fpt_event = False

    for step in xrange(num_steps - 1):
        current_time += tau_override
        # compute propensity functions (alpha) and the partitions for all 12 transitions
        alpha = reaction_propensities_lowmem(current_state, params, fpt_flag=fpt_flag)

        # compute poissonian event counts for all j rxn with param lambda = aj(t) * tau
        fpt_event = calc_rxn_events(current_state, tau_override, alpha, update_vec_array_transpose)

        # update tracking arrays
        if flag_normalize_each_step:
            current_state[:] = [int(current_state[k] + 0.5) if current_state[k] >= 0 else 0
                                for k in xrange(params.numstates)]
        # temp exit and printing
        """
        if step % 10000 == 0:
            print "step", step, "time", current_time, ":", current_state
            if step >= 100000:
                return None, None
        """
        # fpt and establish exit conditions
        if fpt_event:
            assert fpt_flag                                 # just in case, not much cost
            return current_state, current_time
        if establish_flag and current_state[-1] >= params.N:
            return current_state, current_time

    if fpt_flag or establish_flag:  # if code gets here should recursively continue the simulation
        #print "recursing (%.2f) in tauleap lowmem to wait for event flag exit condition" % current_time
        state_end, time_end = stoch_tauleap_lowmem(current_state, num_steps, params, fpt_flag=fpt_flag,
                                                   establish_flag=establish_flag, init_time=current_time)
        return state_end, time_end

    return current_state, current_time


def simulate_dynamics_general(init_cond, times, params, method="libcall"):
    if method == "libcall":
        return ode_libcall(init_cond, times, params)
    elif method == "rk4":
        return ode_rk4(init_cond, times, params)
    elif method == "euler":
        return ode_euler(init_cond, times, params)
    elif method == "gillespie":
        return stoch_gillespie(init_cond, len(times), params)
    elif method == "bnb":
        return stoch_bnb(init_cond, len(times), params)
    elif method == "tauleap":
        return stoch_tauleap(init_cond, len(times), params)
    # TODO also want tau leap brief here i.e. low mem fn and all state high mem fn
    else:
        raise ValueError("method arg invalid, must be one of %s" % SIM_METHODS_VALID)


def fp_from_timeseries(r, sim_method, tol=0.001):
    fp_test = r[-1,:]
    fp_check = r[-2,:]
    if np.linalg.norm(fp_test - fp_check) <= tol:
        return fp_test
    elif sim_method == "gillespie" and np.linalg.norm(fp_test - r[-5,:]) <= 5:  # TODO exit condition for gillespie SS
        return fp_test
    else:
        raise ValueError("timeseries endpoint not a fixed point using dist tol: %.2f" % tol)


def bifurc_value(params, bifurc_name):
    """
    Note: assumes params contains at most one None parameter
    # TODO: implement mu_base (analysis)
    """
    assert params.system == "default"
    p = params
    assert p.mu_base <= 10e-10
    if p.b is not None:
        delta = 1 - p.b
    if p.c is not None:
        s = p.c - 1
    # assumes threshold_2 is stronger constraint, atm hardcode rearrange expression for bifurc param
    if bifurc_name == "bifurc_b":
        delta_val = p.alpha_minus * p.alpha_plus / (s + p.alpha_plus) - (s + p.alpha_minus + p.mu)
        bifurc_val = 1 - delta_val
        return bifurc_val
    elif bifurc_name == "bifurc_c":
        """
        -bifurcation in s = c - 1 occurs at rightmost root of a1 quadratic criterion in general
        -note a1 always has 2 roots for physical parameters
        -note linear a0 criterion has positive slope in s and IS the derivative of a1 wrt s 
         and so its root will always at the midpoint of the two a1 roots
        -need a1 and a0 both positive, since a0 not positive left of its sol and a1 not positive 
         between its roots this implies a1's rightmost root gives the bifurcation point
        """
        poly = np.array([1, delta + p.alpha_plus + p.alpha_minus + p.mu, p.alpha_plus * (delta + p.mu)])
        roots = np.roots(poly)
        s_val = np.max(roots)
        bifurc_val = 1 + s_val
        return bifurc_val
    elif bifurc_name == "bifurc_mu":
        """
        -expect bifurcation in mu to behave similarly to bifurcation in delta (b)
        -this is due to similar functional location in a0, a1 expressions
        """
        mu_option0 = p.alpha_minus * p.alpha_plus / (s + p.alpha_plus) - (s + p.alpha_minus + delta)
        mu_option1 = -(2*s + p.alpha_minus + delta + p.alpha_plus)
        return np.max([mu_option0, mu_option1])
    else:
        raise ValueError(bifurc_name + ' not valid bifurc ID')


def threshold_1(params):
    p = params
    assert p.system == "default"
    assert p.mu_base <= 10e-10
    delta = 1 - p.b
    s = p.c - 1
    return 2 * s + delta + p.alpha_plus + p.alpha_minus + p.mu


def threshold_2(params):
    p = params
    assert p.system == "default"
    assert p.mu_base <= 10e-10
    delta = 1 - p.b
    s = p.c - 1
    return (s + p.alpha_plus) * (s + delta + p.alpha_minus + p.mu) - p.alpha_minus * p.alpha_plus


def q_get(params, sign):
    p = params
    assert p.system == "default"
    assert p.mu_base <= 10e-10
    assert sign in [-1, +1]
    delta = 1 - p.b
    s = p.c - 1
    bterm = p.alpha_plus - p.alpha_minus - p.mu - delta
    return 0.5 / p.alpha_minus * (bterm + sign * np.sqrt(bterm ** 2 + 4 * p.alpha_minus * p.alpha_plus))


def fp_location_noflow(params):
    p = params
    assert p.system == "default"
    q1 = q_get(params, +1)
    q2 = q_get(params, -1)
    assert p.mu_base <= 10e-10
    delta = 1 - p.b
    s = p.c - 1
    conjugate_fps = [[0,0,0], [0,0,0]]
    for idx, q in enumerate([q1,q2]):
        xi = p.N * (s + p.alpha_plus - p.alpha_minus * q) / (s + (delta + s) * q)
        yi = q * xi
        zi = p.N - xi - yi
        conjugate_fps[idx] = [xi, yi, zi]
    return [[0, 0, p.N], conjugate_fps[0], conjugate_fps[1]]


def fp_location_sympy_system(params):
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    sym_x = Symbol("x")
    sym_y = Symbol("y")
    state_vec = [sym_x, sym_y, params.N - sym_x - sym_y]

    # TODO check that new method matches old method
    p = params.mod_copy(params.system_variants(state_vec, None))
    """ OLD METHOD
    if params.system == "feedback_z":
        z = N - sym_x - sym_y
        alpha_plus = alpha_plus * (1 + z**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO*N)**HILL_EXP))
        alpha_minus = alpha_minus * (HILLORIG_Z0_RATIO*N)**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO*N)**HILL_EXP)
    elif params.system == "feedback_yz":
        yz = N - sym_x
        alpha_plus = alpha_plus * (1 + yz**HILL_EXP / (yz**HILL_EXP + (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP))
        alpha_minus = alpha_minus * (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP / (yz**HILL_EXP + (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP)
    elif params.system == "feedback_mu_XZ_model":
        z = N - sym_x - sym_y
        alpha_plus = 0.0
        alpha_minus = 0.0
        mu_base = mu_base * (1 + MUBASE_MULTIPLIER * z**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO * N)**HILL_EXP))
    """

    VV = (p.v_x + p.v_y + p.v_z) / p.N
    xdot = (p.c-p.a)/p.N*sym_x**2 + (p.c-p.b)/p.N*sym_x*sym_y + (p.a-p.c-p.alpha_plus-p.mu_base-VV)*sym_x + p.alpha_minus*sym_y + p.v_x
    ydot = (p.c-p.b)/p.N*sym_y**2 + (p.c-p.a)/p.N*sym_x*sym_y + (p.b-p.c-p.alpha_minus-p.mu-VV)*sym_y + p.alpha_plus*sym_x + p.v_y
    eqns = (xdot, ydot)
    solution = solve(eqns)
    solution_list = [[0,0,0], [0,0,0], [0,0,0]]
    for i in xrange(3):
        x_i = float(re(solution[i][sym_x]))
        y_i = float(re(solution[i][sym_y]))
        solution_list[i] = [x_i, y_i, p.N - x_i - y_i]
    return solution_list


def fp_location_sympy_quartic(params):
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    sym_x = Symbol("x")
    sym_y = Symbol("y")
    state_vec = [sym_x, sym_y, params.N - sym_x - sym_y]

    # TODO check that new method matches old method
    p = params.mod_copy(params.system_variants(state_vec, None))
    """ OLD METHOD
    if params.system == "feedback_z":
        z = N - sym_x - sym_y
        alpha_plus = alpha_plus * (1 + z**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO*N)**HILL_EXP))
        alpha_minus = alpha_minus * (HILLORIG_Z0_RATIO*N)**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO*N)**HILL_EXP)
    elif params.system == "feedback_yz":
        yz = N - sym_x
        alpha_plus = alpha_plus * (1 + yz**HILL_EXP / (yz**HILL_EXP + (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP))
        alpha_minus = alpha_minus * (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP / (yz**HILL_EXP + (HILL_Y0_PLUS_Z0_RATIO*N)**HILL_EXP)
    elif params.system == "feedback_mu_XZ_model":
        z = N - sym_x - sym_y
        alpha_plus = 0.0
        alpha_minus = 0.0
        mu_base = mu_base * (1 + MUBASE_MULTIPLIER * z**HILL_EXP / (z**HILL_EXP + (HILLORIG_Z0_RATIO * N)**HILL_EXP))
    """
    VV = (p.v_x+p.v_y+p.v_z)/p.N
    a0 = (p.c-p.a)/p.N
    a1 = 0.0
    b0 = 0.0
    b1 = (p.c-p.b)/p.N
    c0 = (p.c-p.b)/p.N
    c1 = (p.c-p.a)/p.N
    d0 = (p.a-p.c-p.alpha_plus-p.mu_base-VV)
    d1 = p.alpha_plus
    e0 = p.alpha_minus
    e1 = (p.b-p.c-p.alpha_minus-p.mu-VV)
    f0 = p.v_x
    f1 = p.v_y
    eqn = b1*(a0*sym_x**2 + d0*sym_x + f0)**2 - (c0*sym_x + e0)*(a0*sym_x**2 + d0*sym_x + f0)*(c1*sym_x + e1) + d1*sym_x + f1*(c0*sym_x + e0)**2
    solution = solve(eqn)
    solution_list = [[0,0,0], [0,0,0], [0,0,0]]
    for i in xrange(3):
        x_i = float(re(solution[i]))
        y_i = -(a0*x_i**2 + d0*x_i + f0) / (c0*x_i + e0)  # WARNING TODO: ensure this denom is nonzero
        solution_list[i] = [x_i, y_i, p.N - x_i - y_i]
    return solution_list


def fsolve_func(xvec_guess, params):  # TODO: faster if split into 3 fns w.o if else for feedback cases
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    x0, y0 = xvec_guess
    state_guess = [x0, y0, params.N - x0 - y0]
    p = params.mod_copy(params.system_variants(state_guess, None))
    VV = (p.v_x + p.v_y + p.v_z) / p.N
    xdot = (p.c-p.a)/p.N*x0**2 + (p.c-p.b)/p.N*x0*y0 + (p.a-p.c-p.alpha_plus-p.mu_base-VV)*x0 + p.alpha_minus*y0 + p.v_x
    ydot = (p.c-p.b)/p.N*y0**2 + (p.c-p.a)/p.N*x0*y0 + (p.b-p.c-p.alpha_minus-p.mu-VV)*y0 + p.alpha_plus*x0 + p.v_y
    return [xdot, ydot]


def fp_location_fsolve(params, check_near_traj_endpt=True, gridsteps=15, tol=10e-4, buffer=False):
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    N = params.N
    unique_solutions = []
    # first check for roots near trajectory endpoints (possible stable roots)
    if check_near_traj_endpt:
        init_cond = INIT_COND
        times = np.linspace(TIME_START, TIME_END, NUM_STEPS + 1)
        traj, _ = simulate_dynamics_general(init_cond, times, params, method="libcall")
        fp_guess = traj[-1][0:2]
        solution, infodict, _, _ = fsolve(fsolve_func, fp_guess, (params), full_output=True)
        if np.linalg.norm(infodict["fvec"]) <= 10e-3:  # only append actual roots (i.e. f(x)=0)
            unique_solutions.append([solution[0], solution[1], N - solution[0] - solution[1]])
    # grid search of solution space (positive simplex):
    if buffer:
        buffer = int(gridsteps/5)
        irange = range(-buffer, gridsteps + buffer)
        jrange = range(-buffer, gridsteps + buffer)
    else:
        irange = range(gridsteps)
        jrange = range(gridsteps)
    for i in irange:
        x_guess = N * i / float(gridsteps)
        for j in jrange:
            y_guess = N * j / float(gridsteps)
            # TODO: this returns jacobian estimate.. use it
            solution, infodict, _, _ = fsolve(fsolve_func, [x_guess, y_guess], (params), full_output=True)
            append_flag = True
            for k, usol in enumerate(unique_solutions):
                if np.abs(solution[0] - usol[0]) <= tol:   # only store unique roots from list of all roots
                    append_flag = False
                    break
            if append_flag:
                if np.linalg.norm(infodict["fvec"]) <= 10e-3:    # only append actual roots (i.e. f(x)=0)
                    unique_solutions.append([solution[0], solution[1], N - solution[0] - solution[1]])
    return unique_solutions


def fp_location_general(params, solver_fsolve=True, solver_fast=False, solver_explicit=False):
    # TODO: sympy solver often fails when feedback added in
    # TODO: cleanup the flags here
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    if solver_fsolve:
        return fp_location_fsolve(params)
    elif solver_explicit:
        assert params.system == "default"
        return fp_location_noflow(params)
    else:
        if solver_fast:
            return fp_location_sympy_quartic(params)
        else:
            return fp_location_sympy_system(params)


def jacobian_3d(params, fp):
    p = params
    assert p.system == "default"
    assert p.mu_base <= 10e-10
    M = np.array([[p.a - p.alpha_plus - p.mu_base, p.alpha_minus, 0],
                  [p.alpha_plus, p.b - p.alpha_minus - p.mu, 0],
                  [p.mu_base, p.mu, p.c]])
    x, y, z = fp
    diag = p.a*x + p.b*y + p.c*z + p.v_x + p.v_y + p.v_z
    r1 = [diag + x*p.a, x*p.b, x*p.c]
    r2 = [y*p.a, diag + y*p.b, y*p.c]
    r3 = [z*p.a, z*p.b, diag + z*p.c]
    return M - 1/p.N*np.array([r1,r2,r3])


def jacobian_numerical_2d(params, fp):
    # TODO: can use numdifftools jacobian function instead
    # TODO: move scope of func xdot etc up and use them both in func fsolve
    assert params.system in ["default", "feedback_z", "feedback_yz"]
    def func_xdot(fp):
        x, y = fp[0], fp[1]
        state_vec = [x, y, params.N - x - y]
        p = params.mod_copy(params.system_variants(state_vec, None))
        VV = (p.v_x + p.v_y + p.v_z) / p.N
        return (p.c - p.a) / p.N * x ** 2 + (p.c - p.b) / p.N * x * y + (p.a - p.c - p.alpha_plus - p.mu_base - VV) * x \
               + p.alpha_minus * y + p.v_x
    def func_ydot(fp):
        x, y = fp[0], fp[1]
        state_vec = [x, y, params.N - x - y]
        p = params.mod_copy(params.system_variants(state_vec, None))
        VV = (p.v_x + p.v_y + p.v_z) / p.N
        return (p.c-p.b)/p.N*y**2 + (p.c-p.a)/p.N*x*y + (p.b-p.c-p.alpha_minus-p.mu-VV)*y + p.alpha_plus*x + p.v_y
    epsilon = 10e-4
    row_x = approx_fprime(fp, func_xdot, epsilon)
    row_y = approx_fprime(fp, func_ydot, epsilon)
    return np.array([row_x, row_y])


def is_stable(params, fp, method="numeric_2d"):
    if method == "numeric_2d":
        assert len(fp) == 2
        J = jacobian_numerical_2d(params, fp)
        eigenvalues, V = np.linalg.eig(J)
    elif method == "algebraic_3d":
        J = jacobian_3d(params, fp)
        eigenvalues, V = np.linalg.eig(J)
    else:
        raise ValueError("method must be 'numeric_2d' or 'algebraic_3d'")
    return all(eig < 0 for eig in eigenvalues)


def get_fp_stable_and_not(params):
    fp_locs = fp_location_general(params, solver_fsolve=True)
    fp_locs_stable = []
    fp_locs_unstable = []
    for fp in fp_locs:
        if is_stable(params, fp[0:2], method="numeric_2d"):
            fp_locs_stable.append(fp)
            # eigs,V = np.linalg.eig(jacobian_numerical_2d(params, fp[0:2], ode_system))
            # print fp, eigs
        else:
            fp_locs_unstable.append(fp)
    return fp_locs_stable, fp_locs_unstable


def get_physical_fp_stable_and_not(params, verbose=False):
    fp_locs = fp_location_general(params, solver_fsolve=True)
    fp_locs_physical_stable = []
    fp_locs_physical_unstable = []
    for fp in fp_locs:
        if all([val > -1e-4 for val in fp]):
            if is_stable(params, fp[0:2], method="numeric_2d"):
                fp_locs_physical_stable.append(fp)
                #eigs,V = np.linalg.eig(jacobian_numerical_2d(params, fp[0:2], ode_system))
                #print fp, eigs
            else:
                fp_locs_physical_unstable.append(fp)

    if verbose:
        print "\nFP NOTES for b,c", params.b, params.c
        print "ALL FP: (%d)" % len(fp_locs)
        for fp in fp_locs:
            print fp
        print "PHYS/STABLE FP: (%d)" % len(fp_locs_physical_stable)
        for fp in fp_locs_physical_stable:
            print fp
        if len(fp_locs_physical_stable) == 0:
            print "WARNING: 0 phys and stable FP"
            init_cond = INIT_COND
            times = np.linspace(TIME_START, TIME_END, NUM_STEPS + 1)
            traj, _ = simulate_dynamics_general(init_cond, times, params, method="libcall")
            fp_guess = traj[-1][:]
            print "FP from traj at all-x is:"
            print fp_guess
            print "ode_system_vector at possible FP is"
            print params.ode_system_vector(fp_guess, None)

    return fp_locs_physical_stable, fp_locs_physical_unstable


def random_init_cond(params):
    N = float(params.N)
    init_cond = np.zeros(params.numstates)
    for idx in xrange(params.numstates - 1):
        init_cond[idx] = (N - np.sum(init_cond)) * np.random.random_sample()
    init_cond[-1] = N - np.sum(init_cond)
    return list(init_cond)


def map_init_name_to_init_cond(params, init_name):
    N = int(params.N)
    if params.numstates == 3:
        init_map = {"x_all": [N, 0, 0],
                    "z_all": [0, 0, N],
                    "mixed": [int(0.7 * N), int(0.2 *N ), int(0.1 * N)],
                    "midpoint": [N/3, N/3, N - 2*N/3],
                    "z_close": [int(N*0.05), int(N*0.05), int(N*0.9)],
                    "random": random_init_cond(params)}
    elif params.numstates == 2:
        init_map = {"x_all": [N, 0],
                    "z_all": [0, N],
                    "mixed": [int(0.8 * N), int(0.2 * N)],
                    "midpoint": [N/2, N/2],
                    "z_close": [int(N*0.1), int(N*0.9)],
                    "random": random_init_cond(params)}
    elif params.numstates == 4:
        init_map = {"x_all": [N, 0, 0, 0],
                    "z_all": [0, 0, N, 0],
                    "mixed": [int(0.7 * N), int(0.2 * N), int(0.1 * N), 0],
                    "midpoint": [N/3, N/3, N - 2*N/3, 0],
                    "z_close": [int(N*0.05), int(N*0.05), int(N*0.9), 0],
                    "random": random_init_cond(params)}
    else:
        init_map = None
    return init_map[init_name]
