import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd
import seaborn as sns; sns.set()

from custom_rbm import RBM_gaussian_custom
from plotting import image_fancy
from RBM_train import load_rbm_hopfield, TRAINING
from RBM_assess import get_X_y_dataset
from settings import DIR_OUTPUT, DIR_MODELS


def esimate_logZ_with_AIS(weights, field_visible, field_hidden, beta=1.0, num_chains=100, num_steps=1000):
    # Note: without fields, for hopfield or generic bg-RBM
    #   the ising/binary quadratic form is pos. semi. def and so all the boltzmann factors are >= 1 (=1 when E=0)
    #   in the limit beta -> 0 the distribution approaches uniform distr. with 2^N terms of boltzmann weights 1
    #   this means ln Z is bounded below by ln 2^N ~ 543.427
    # Note: doing 9 'runs' of this for 1000 steps was ~1.5hr
    # Run num_chains AIS chains in parallel for num_steps
    N = field_visible.shape[0]
    p = field_hidden.shape[0]
    assert weights.shape[0] == N
    assert weights.shape[1] == p

    dims = p
    dtype = tf.float32

    # fix target model  # TODO minor speedup if we move out
    weights = tf.convert_to_tensor(weights, dtype=dtype)
    field_visible = tf.convert_to_tensor(field_visible, dtype=dtype)
    field_hidden = tf.convert_to_tensor(field_hidden, dtype=dtype)

    # define proposal distribution
    tfd = tfp.distributions
    proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([dims], dtype=dtype))
    proposal_log_prob_fn = proposal.log_prob

    # define target distribution
    target_log_prob_const = N * np.log(2.0) - (dims / 2.0) * np.log(2.0 * np.pi / beta)

    def target_log_prob_fn(hidden_states):
        # given vector size N ints, return scalar for each chain
        fvals = [0.0] * len(hidden_states)
        # TODO tensor speedup test with p > 1
        for idx, hidden in enumerate(hidden_states):

            hidden_to_sqr = hidden - field_hidden
            term1 = tf.tensordot(hidden_to_sqr, hidden_to_sqr, 1)

            cosh_arg = beta * (tf.tensordot(weights, hidden, 1) + field_visible)
            log_cosh_vec = tf.math.log(tf.math.cosh(cosh_arg))
            term2 = tf.math.reduce_sum(log_cosh_vec)

            fvals[idx] = - (beta / 2.0) * term1 + term2
        fvals = tf.convert_to_tensor(fvals, dtype=dtype) + target_log_prob_const   # TODO this const can be removed and added at the end (speedup)
        return fvals

    # draw 100 samples from the proposal distribution
    init_state = proposal.sample(num_chains)

    chains_state, ais_weights, kernels_results = (
        tfp.mcmc.sample_annealed_importance_chain(
            num_steps=num_steps,
            proposal_log_prob_fn=proposal_log_prob_fn,
            target_log_prob_fn=target_log_prob_fn,
            current_state=init_state,
            make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=tlp_fn,
                step_size=0.2,
                num_leapfrog_steps=2)))

    log_Z = (tf.reduce_logsumexp(ais_weights) - np.log(num_chains))
    return log_Z.numpy(), chains_state


def esimate_logZ_with_reverse_AIS_algo2(rbm, weights, field_visible, field_hidden, beta=1.0, num_chains=100, num_steps=1000, CDK=100):
    # Follows https://www.cs.cmu.edu/~rsalakhu/papers/RAISE.pdf Algorithm 2
    # TODO compare vs algorithm 3, need to implement
    N = field_visible.shape[0]
    p = field_hidden.shape[0]
    assert weights.shape[0] == N
    assert weights.shape[1] == p

    dims = p
    dtype = tf.float32

    # fix target model  # TODO minor speedup if we move out
    weights = tf.convert_to_tensor(weights, dtype=dtype)
    field_visible = tf.convert_to_tensor(field_visible, dtype=dtype)
    field_hidden = tf.convert_to_tensor(field_hidden, dtype=dtype)

    # define proposal distribution
    tfd = tfp.distributions
    target = tfd.MultivariateNormalDiag(loc=tf.zeros([dims], dtype=dtype))
    target_log_prob_fn = target.log_prob

    # define target distribution
    proposal_log_prob_const = N * np.log(2.0) - (dims / 2.0) * np.log(2.0 * np.pi / beta)

    def proposal_log_prob_fn(hidden_states):
        # given vector size N ints, return scalar for each chain
        fvals = [0.0] * len(hidden_states)
        # TODO tensor speedup test with p > 1
        for idx, hidden in enumerate(hidden_states):

            hidden_to_sqr = hidden - field_hidden
            term1 = tf.tensordot(hidden_to_sqr, hidden_to_sqr, 1)

            cosh_arg = beta * (tf.tensordot(weights, hidden, 1) + field_visible)
            log_cosh_vec = tf.math.log(tf.math.cosh(cosh_arg))
            term2 = tf.math.reduce_sum(log_cosh_vec)

            fvals[idx] = - (beta / 2.0) * term1 + term2
        fvals = tf.convert_to_tensor(fvals, dtype=dtype) + proposal_log_prob_const
        return fvals

    # draw 100 samples from the proposal distribution
    stdev = np.sqrt(1.0 / beta)
    hightemp_zone = CDK / 2  # first CDK/2 steps will be high temp
    high_beta = 0.5
    high_stdev = np.sqrt(1.0 / high_beta)
    def proposal_sampler(num_chains, CDK=CDK):
        # TODO if we believe the batch CD-K samples, they can be passed as initial proposal sample
        # 0) generate num_chains visible states at random, uniform p=0.5 spin flip (TODO we may not want 50% on/off)
        vis_chains_bool = np.random.uniform(size=(num_chains, N))
        vis_chains = vis_chains_bool * 2 - 1
        # 1) do CD-K to get approximate sample from RBM
        #    NOTE need torch tensor k x 784 to use existing CD module
        visible_sampled_TORCH = torch.from_numpy(vis_chains).float()

        for step in range(CDK):
            if step < hightemp_zone:
                hidden_sampled_TORCH = rbm.sample_hidden(visible_sampled_TORCH, stdev=high_stdev)
                visible_sampled_TORCH = rbm.sample_visible(hidden_sampled_TORCH, beta=high_beta)
            else:
                hidden_sampled_TORCH = rbm.sample_hidden(visible_sampled_TORCH, stdev=stdev)
                visible_sampled_TORCH = rbm.sample_visible(hidden_sampled_TORCH, beta=beta)
        # 2) take final hidden states as the sample
        hidden_sampled_TF = tf.convert_to_tensor( hidden_sampled_TORCH.numpy() )
        return hidden_sampled_TF
    init_state = proposal_sampler(num_chains)

    chains_state, ais_weights, kernels_results = (
        tfp.mcmc.sample_annealed_importance_chain(
            num_steps=num_steps,
            proposal_log_prob_fn=proposal_log_prob_fn,
            target_log_prob_fn=target_log_prob_fn,
            current_state=init_state,
            make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=tlp_fn,
                step_size=0.2,
                num_leapfrog_steps=2)))

    log_Z = -1 * (tf.reduce_logsumexp(ais_weights) - np.log(num_chains))  # mult -1 because estimator for Z^(-1)

    return log_Z.numpy(), chains_state


def AIS_update_visible_numpy(hidden_state, weights, beta, N, alpha):
    # vectorized to operate on chain
    visible_activations = beta * alpha * np.dot(hidden_state, weights.T)
    sigmoid_arg = -2 * visible_activations
    visible_probabilities = 1 / (1 + np.exp(sigmoid_arg))
    visible_sampled = np.random.binomial(1, visible_probabilities, size=(nchains, N))
    visible_sampled_phys = -1 + visible_sampled * 2
    return visible_sampled_phys


def AIS_update_hidden_numpy(visible_state, weights, stdev, alpha):
    # vectorized to operate on chain
    hidden_activations = alpha * np.dot(visible_state, weights)
    hidden_sampled = np.random.normal(hidden_activations, stdev)
    return hidden_sampled


def compute_log_f_k_hidden(chain_state, weights, alpha):
    assert len(chain_state.shape) == 1  # TODO vectorize so log_f_k is size N_chains ?
    # un-normalized prob for the intermediate distribution with param alpha_k
    lambda_sqr = np.dot(chain_state, chain_state)                              # TODO vectorize
    ln_cosh_vec = np.log( np.cosh(
        alpha * beta * np.dot(weights, chain_state)
        ) )  # TODO vectorize
    log_f_k = -beta * lambda_sqr / 2 + np.sum(ln_cosh_vec)                     # TODO vectorize
    return log_f_k


def compute_log_f_k_joint(chain_visible, chain_hidden, weights, alpha):
    assert len(chain_visible.shape) == 1  # TODO vectorize so log_f_k is size N_chains ?
    assert len(chain_hidden.shape) == 1  # TODO vectorize so log_f_k is size N_chains ?
    # un-normalized prob for the intermediate distribution with param alpha_k
    lambda_sqr = np.dot(chain_hidden, chain_hidden)                              # TODO vectorize

    dot_W_h = np.dot(weights, chain_hidden)
    term2 = np.dot(chain_visible, dot_W_h)

    log_f_k_joint = -beta * lambda_sqr / 2 + beta * alpha * term2                     # TODO vectorize
    return log_f_k_joint


def manual_AIS(rbm, beta, nchains=100, nsteps=10, CDK=1):
    np.random.seed()
    stepper = 'gibbs'  # MCMC_MH or gibbs

    # Note 1: MCMC will be gibbs chain as suggest in RAISE paper
    # Note 2: initial distribution unit normal is like RBM with W_i\mu = 0 (W=0 weights)

    N = rbm.num_visible
    p = rbm.num_hidden
    stdev = np.sqrt(1/beta)

    log_z_prefactor = N * np.log(2) - p/2.0 * np.log(2 * np.pi / beta)

    weights_numpy = rbm.weights.numpy()
    assert torch.allclose(rbm.visible_bias, torch.zeros(N))
    assert torch.allclose(rbm.hidden_bias, torch.zeros(p))

    # STEP 1: initial samples
    init_chain_hidden = np.random.normal(loc=0.0, scale=stdev, size=(nchains, p))

    # STEP 2: initial AIS weights
    Z_init = (2 * np.pi / beta) ** (p/2)
    log_ais_weights = np.zeros(nchains) + np.log(Z_init)

    # STEP 3: perform iteration over intermediate distributions f_k( )
    chain_state_hidden = init_chain_hidden
    chain_state_visible = np.zeros((nchains, N))
    log_f_k_numerator = np.zeros(nchains)
    log_f_k_denominator = np.zeros(nchains)
    alpha_k = np.linspace(0, 1, nsteps + 1)

    for k in range(1, nsteps + 1):

        if k % 100 == 0:
            print('step', k)

        alpha_current = alpha_k[k]
        alpha_prev = alpha_k[k - 1]

        # update ais_weights
        for c in range(nchains):
            # TODO vectorize
            log_f_k_numerator[c] = compute_log_f_k_hidden(chain_state_hidden[c, :], weights_numpy, alpha_current)
            log_f_k_denominator[c] = compute_log_f_k_hidden(chain_state_hidden[c, :], weights_numpy, alpha_prev)
            log_ais_weights[c] = log_ais_weights[c] + log_f_k_numerator[c] - log_f_k_denominator[c]
            #if k % 100 == 0:
            #   print('step', k, 'chain', c, '|||', log_f_k_numerator[c], log_f_k_denominator[c], np.exp(log_f_k_numerator[c] - log_f_k_denominator[c]), log_ais_weights[c], np.exp(log_ais_weights[c]))

        # update chain
        if stepper == 'MCMC_MH':
            # refer to: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
            # have x_t;
            # sample x' from proposal distribution N(0, beta_inv I) (better if symmetric)
            X_candidate = np.random.normal(loc=0.0, scale=stdev, size=(nchains, p))
            # get f(x')
            log_f_candidates = np.zeros(nchains)
            for c in range(nchains):  # TODO loop above if it works better? else H-AIS
                log_f_candidates[c] = compute_log_f_k_hidden(X_candidate[c, :], alpha_current)
                # get U[0,1] r.v. to compare vs acceptance ratio
                uu = np.random.rand()
                ratio = log_f_candidates[c] / compute_log_f_k_hidden(chain_state_hidden[c, :], alpha_current)
                if uu <= ratio:
                    chain_state_hidden[c, :] = X_candidate[c, :]
        else:
            for samplestep in range(CDK):
                chain_state_visible = AIS_update_visible_numpy(chain_state_hidden, weights_numpy, beta, N, alpha_current)
                chain_state_hidden = AIS_update_hidden_numpy(chain_state_visible, weights_numpy, stdev, alpha_current)

    # STEP 4: return estimate
    Z_est = np.sum(np.exp(log_ais_weights)) / nchains  # Note TF has log-sum-exp fn they claim more stable (how?)
    log_Z_est = np.log(Z_est)
    log_Z_total = log_Z_est + log_z_prefactor

    return log_Z_total, chain_state_hidden


def manual_AIS_reverse(rbm, beta, test_cases, nchains=100, nsteps=10, CDK=1):
    # TODO why is p_ann (output of their algorithm) > 1?
    N = rbm.num_visible
    p = rbm.num_hidden
    stdev = np.sqrt(1/beta)

    ntest = test_cases.shape[0]
    assert N == test_cases.shape[1]

    weights_numpy = rbm.weights.numpy()
    assert torch.allclose(rbm.visible_bias, torch.zeros(N))
    assert torch.allclose(rbm.hidden_bias, torch.zeros(p))

    def compute_log_f_k_spinsOnly(chain_state_visible):
        assert len(chain_state_visible.shape) == 1  # TODO vectorize so log_f_k is size N_chains ?
        # just need scaled Ising energy which is + beta/2 * ||s.T W||^2
        dot_s_W = np.dot(chain_state_visible, weights_numpy)

        plt.plot(dot_s_W); plt.show(); plt.close()

        energy_unscaled = np.dot(dot_s_W, dot_s_W)
        energy_scaled = 0.5 * beta * energy_unscaled
        return energy_scaled

    # definitions
    log_p_annealed_estimates = np.zeros(ntest)
    log_Z_est_integral = np.zeros(ntest)

    alpha_k = np.linspace(0, 1, nsteps + 1)
    log_z_prefactor = - p / 2.0 * np.log(2 * np.pi / beta) # + N * np.log(2)  # TODO care we use f_k_joint so no need to add that part of pre-factor to Z at the end
    log_Z_init = N * np.log(2) - (p / 2) * np.log(2 * np.pi / beta)  # TODO make sure adding N log 2 and sign correct

    for test_idx, visible_test in enumerate(test_cases):  # TODO ensure visible_test is standard vector shaped np array 1D
        print("RAISE: test sample #%d" % test_idx)

        # 1) sample M h_test samples from RBM distribution
        chain_visible = np.array([visible_test] * nchains)
        chain_hidden = AIS_update_hidden_numpy(chain_visible, weights_numpy, stdev, 1.0)

        # 2) initialize reverse AIS weights
        log_w_init_numerator = compute_log_f_k_spinsOnly(visible_test)
        log_w_init = log_w_init_numerator - log_Z_init
        print("log_w_init", log_w_init, log_w_init_numerator, log_Z_init)
        log_ais_weights = np.zeros(nchains) + log_w_init

        # 3) loop over AIS steps
        log_f_k_numerator = np.zeros(nchains)
        log_f_k_denominator = np.zeros(nchains)
        for step, k in enumerate(range(nsteps - 1, -1, -1)):

            #if step % 100 == 0:
            #    print('step', step, 'kval', k)
            alpha_current = alpha_k[k]
            alpha_prev = alpha_k[k + 1]  # note sign flip here, "previous" means higher k value

            # update chain
            for samplestep in range(CDK):
                chain_visible = AIS_update_visible_numpy(chain_hidden, weights_numpy, beta, N, alpha_current)
                chain_hidden = AIS_update_hidden_numpy(chain_visible, weights_numpy, stdev, alpha_current)

            # update ais_weights
            for c in range(nchains):
                # TODO vectorize
                # Get f_k from hidden alone
                #log_f_k_numerator[c] = compute_log_f_k_hidden(chain_hidden[c, :], weights_numpy, alpha_current)  # TODO compare with f(s,h) form
                #log_f_k_denominator[c] = compute_log_f_k_hidden(chain_hidden[c, :], weights_numpy, alpha_prev)   # TODO compare with f(s,h) form

                # Get f_k from hidden, visible (joint)
                log_f_k_numerator[c] = compute_log_f_k_joint(chain_visible[c, :], chain_hidden[c, :], weights_numpy, alpha_current)
                log_f_k_denominator[c] = compute_log_f_k_joint(chain_visible[c, :], chain_hidden[c, :], weights_numpy, alpha_prev)

                log_ais_weights[c] = log_ais_weights[c] + log_f_k_numerator[c] - log_f_k_denominator[c]

                # prints
                #if step % 100 == 0:
                #   print('test', test_idx, 'step', step, 'kval', k, 'chain', c, '|||', log_f_k_numerator[c], log_f_k_denominator[c], np.exp(log_f_k_numerator[c] - log_f_k_denominator[c]), log_ais_weights[c], np.exp(log_ais_weights[c]))
                if c == 4:
                    print('test', test_idx, 'step', step, 'kval', k, 'chain', c, '|||', log_f_k_numerator[c],
                          log_f_k_denominator[c], np.exp(log_f_k_numerator[c] - log_f_k_denominator[c]), log_ais_weights[c],
                          np.exp(log_ais_weights[c]))

        # 4) return estimate for the test case
        log_p_annealed_estimates[test_idx] = np.log(np.sum(np.exp(log_ais_weights))) - np.log(nchains)  # TODO more stable version of Log-Sum-Exp ?
        log_Z_est_integral[test_idx] = log_w_init_numerator - log_p_annealed_estimates[test_idx]
        print("RAISE: test sample #%d" % test_idx, '|||', log_ais_weights, np.log(np.sum(np.exp(log_ais_weights))), np.log(nchains), log_w_init_numerator)

    # Average the estimates (over test cases) to get est for log_Z
    log_Z_est = np.sum(log_Z_est_integral) / ntest
    log_Z_total = log_Z_est + log_z_prefactor

    # plot hists for debugging
    plt.hist(log_Z_est_integral)
    plt.title('log Z_integral')
    plt.show(); plt.close()
    plt.hist(log_p_annealed_estimates)
    plt.title('log p_ann')
    plt.show(); plt.close()

    return log_Z_total, chain_hidden


def manual_AIS_reverse_algo3(rbm, beta, test_cases, nchains=100, nsteps=10, CDK=1):
    N = rbm.num_visible
    p = rbm.num_hidden
    stdev = np.sqrt(1/beta)

    ntest = test_cases.shape[0]
    assert N == test_cases.shape[1]

    weights_numpy = rbm.weights.numpy()
    assert torch.allclose(rbm.visible_bias, torch.zeros(N))
    assert torch.allclose(rbm.hidden_bias, torch.zeros(p))

    def compute_log_f_k_spinsOnly(chain_state_visible):
        assert len(chain_state_visible.shape) == 1  # TODO vectorize so log_f_k is size N_chains ?
        # just need scaled Ising energy which is + beta/2 * ||s.T W||^2
        dot_s_W = np.dot(chain_state_visible, weights_numpy)

        #plt.plot(dot_s_W); plt.show(); plt.close()

        energy_unscaled = np.dot(dot_s_W, dot_s_W)
        energy_scaled = 0.5 * beta * energy_unscaled
        return energy_scaled

    # definitions
    log_p_annealed_estimates = np.zeros(ntest)
    log_Z_est_integral = np.zeros(ntest)

    alpha_k = np.linspace(0, 1, nsteps + 1)
    log_z_prefactor = N * np.log(2) - p / 2.0 * np.log(2 * np.pi / beta)
    #Z_init = (2 * np.pi / beta) ** (p / 2)
    log_Z_init = N * np.log(2) - (p / 2) * np.log(2 * np.pi / beta)  # TODO make sure adding N log 2 and sign correct


    for test_idx, visible_test in enumerate(test_cases):  # TODO ensure visible_test is standard vector shaped np array 1D
        print("RAISE: test sample #%d" % test_idx)

        # 1) sample M h_test samples from "0" RBM distribution
        chain_visible = np.array([visible_test] * nchains)
        chain_hidden = np.random.normal(loc=0.0, scale=stdev, size=(nchains, p))

        # 2) initialize reverse AIS weights (uniform distribution on N spins)
        #log_w_init_numerator = compute_log_f_k_spinsOnly(visible_test)
        log_w_init = - N * np.log(2)
        print("log_w_init", log_w_init)
        log_ais_weights = np.zeros(nchains) + log_w_init

        # 3) loop over AIS steps
        log_f_k_numerator = np.zeros(nchains)
        log_f_k_denominator = np.zeros(nchains)

        # =================================
        # LOOP A
        # =================================
        for step, k in enumerate(range(1, nsteps + 1)):

            #if step % 100 == 0:
            #    print('step', step, 'kval', k)
            alpha_current = alpha_k[k]
            alpha_prev = alpha_k[k - 1]  # note sign

            # update chain (keep v fixed; only update h)
            for samplestep in range(CDK):
                #chain_visible = AIS_update_visible_numpy(chain_hidden, weights_numpy, beta, N, alpha_current)
                chain_hidden = AIS_update_hidden_numpy(chain_visible, weights_numpy, stdev, alpha_current)

            # update ais_weights
            for c in range(nchains):
                # TODO vectorize
                # Get f_k from hidden alone
                #log_f_k_numerator[c] = compute_log_f_k_hidden(chain_hidden[c, :], weights_numpy, alpha_current)  # TODO compare with f(s,h) form
                #log_f_k_denominator[c] = compute_log_f_k_hidden(chain_hidden[c, :], weights_numpy, alpha_prev)   # TODO compare with f(s,h) form

                # Get f_k from hidden, visible (joint)
                log_f_k_numerator[c] = compute_log_f_k_joint(chain_visible[c, :], chain_hidden[c, :], weights_numpy, alpha_current)
                log_f_k_denominator[c] = compute_log_f_k_joint(chain_visible[c, :], chain_hidden[c, :], weights_numpy, alpha_prev)

                log_ais_weights[c] = log_ais_weights[c] + log_f_k_numerator[c] - log_f_k_denominator[c]
                #if step % 100 == 0:
                #   print('test', test_idx, 'step', step, 'kval', k, 'chain', c, '|||', log_f_k_numerator[c], log_f_k_denominator[c], np.exp(log_f_k_numerator[c] - log_f_k_denominator[c]), log_ais_weights[c], np.exp(log_ais_weights[c]))
                if c == 4:
                    print('test', test_idx, 'step', step, 'kval', k, 'chain', c, '|||', log_f_k_numerator[c],
                          log_f_k_denominator[c], np.exp(log_f_k_numerator[c] - log_f_k_denominator[c]), log_ais_weights[c],
                          np.exp(log_ais_weights[c]))

        # =================================
        # LOOP B
        # =================================
        for step, k in enumerate(range(nsteps - 1, -1, -1)):

            # if step % 100 == 0:
            #    print('step', step, 'kval', k)
            alpha_current = alpha_k[k]
            alpha_prev = alpha_k[k + 1]  # note sign flip here, "previous" means higher k value

            # update chain
            for samplestep in range(CDK):
                chain_visible = AIS_update_visible_numpy(chain_hidden, weights_numpy, beta, N, alpha_current)
                chain_hidden = AIS_update_hidden_numpy(chain_visible, weights_numpy, stdev, alpha_current)

            # update ais_weights
            for c in range(nchains):
                # TODO vectorize
                # Get f_k from hidden alone
                #log_f_k_numerator[c] = compute_log_f_k_hidden(chain_hidden[c, :], alpha_current)  # TODO compare with f(s,h) form
                #log_f_k_denominator[c] = compute_log_f_k_hidden(chain_hidden[c, :], alpha_prev)  # TODO compare with f(s,h) form

                # Get f_k from hidden, visible (joint)
                log_f_k_numerator[c] = compute_log_f_k_joint(chain_visible[c, :], chain_hidden[c, :], weights_numpy, alpha_current)
                log_f_k_denominator[c] = compute_log_f_k_joint(chain_visible[c, :], chain_hidden[c, :], weights_numpy, alpha_prev)

                log_ais_weights[c] = log_ais_weights[c] + log_f_k_numerator[c] - log_f_k_denominator[c]
                # if step % 100 == 0:
                #   print('test', test_idx, 'step', step, 'kval', k, 'chain', c, '|||', log_f_k_numerator[c], log_f_k_denominator[c], np.exp(log_f_k_numerator[c] - log_f_k_denominator[c]), log_ais_weights[c], np.exp(log_ais_weights[c]))
                if c == 4:
                    print('test', test_idx, 'step', step, 'kval', k, 'chain', c, '|||', log_f_k_numerator[c],
                          log_f_k_denominator[c], np.exp(log_f_k_numerator[c] - log_f_k_denominator[c]),
                          log_ais_weights[c],
                          np.exp(log_ais_weights[c]))

        # 4) return estimate for the test case
        energy_vtest = compute_log_f_k_spinsOnly(visible_test)
        log_p_annealed_estimates[test_idx] = np.log(np.sum(np.exp(log_ais_weights))) - np.log(nchains)  # TODO more stable version of Log-Sum-Exp ?
        log_Z_est_integral[test_idx] = energy_vtest - log_p_annealed_estimates[test_idx]
        print("RAISE: test sample #%d" % test_idx, '|||', log_ais_weights, np.log(np.sum(np.exp(log_ais_weights))), np.log(nchains), energy_vtest)

    # Average the estimates (over test cases) to get est for log_Z
    log_Z_est = np.sum(log_Z_est_integral) / ntest
    log_Z_total = log_Z_est + log_z_prefactor

    # plot hists for debugging
    plt.hist(log_Z_est_integral)
    plt.title('log Z_integral')
    plt.show(); plt.close()
    plt.hist(log_p_annealed_estimates)
    plt.title('log p_ann')
    plt.show(); plt.close()

    return log_Z_total, chain_hidden


def subsample_test_cases(data_samples, ntest):
    tot_samples = data_samples.shape[0]
    N = data_samples.shape[1]

    indices = np.random.choice(tot_samples, ntest)  # select indices randomly
    print("indices", indices)
    test_cases = data_samples[indices, :]

    return test_cases


def get_obj_term_A(dataset_prepped, weights, field_visible, field_hidden, beta=1.0):
    data_indep_term = - (beta / 2.0) * np.dot(field_hidden, field_hidden)

    data_sum = 0
    for x in dataset_prepped:
        A = np.dot(field_visible, x)
        Bvec = field_hidden + np.dot(weights.T, x)
        data_sum += A + 0.5 * np.dot(Bvec, Bvec)

    val_avg = beta * data_sum / len(dataset_prepped) + data_indep_term
    return val_avg


def chain_state_to_images(chains_numpy, rbm, num_images, beta=200.0):
    num_chains = chains_numpy.shape[0]
    print(chains_numpy.shape)

    for idx in range(num_images):
        hidden_state_torch = torch.from_numpy(chains_numpy[idx, :]).float()
        visible_state_torch = rbm.sample_visible(hidden_state_torch, beta=beta)  # i.e. sample with zero noise
        visible_state_numpy = visible_state_torch.numpy()
        image = visible_state_numpy.reshape((28,28))
        plt.figure()
        image_fancy(image, ax=plt.gca(), show_labels=False)
        plt.show(); plt.close()

    return


if __name__ == '__main__':

    generate_data = False
    specific_check = True
    compare_methods = False

    if generate_data:

        # AIS settings
        steps = 200  #500  # 1000 and 5000 similar, very slow

        # prep dataset
        training_subsample = TRAINING[:]
        X, _ = get_X_y_dataset(training_subsample, dim_visible=28**2, binarize=True)

        k_list = [1,2,3,4,5,6,7,8,9,10]

        for k_patterns in k_list:
            # load model
            fname = 'hopfield_mnist_%d0.npz' % k_patterns
            rbm = load_rbm_hopfield(npzpath=DIR_MODELS + os.sep + 'saved' + os.sep + fname)
            weights = rbm.internal_weights
            visible_field = rbm.visible_field
            hidden_field = rbm.hidden_field

            # get loss terms (simple term and logZ)
            runs = 2
            beta_list = np.linspace(0.5, 10, 20).astype(np.float32)
            #beta_list = np.linspace(2, 4, 3).astype(np.float32)
            #beta_list = np.linspace(60, 200, 30).astype(np.float32)
            termA_arr = np.zeros((runs, len(beta_list)))
            logZ_arr = np.zeros((runs, len(beta_list)))
            score_arr = np.zeros((runs, len(beta_list)))

            for idx in range(len(beta_list)):
                beta = beta_list[idx]
                obj_term_A = get_obj_term_A(X, weights, visible_field, hidden_field, beta=beta)
                for k in range(runs):
                    termA_arr[k, idx] = obj_term_A  # still keep duplicate values for the plot emphasis
                    #print('computing loss term B (ln Z)')
                    logZ, _ = esimate_logZ_with_AIS(weights, visible_field, hidden_field, beta=beta, num_steps=steps)
                    score = obj_term_A - logZ
                    print('mean log p(data):', score, '(run %d, beta=%.2f, A=%.2f, B=%.2f)' % (k, beta, obj_term_A, logZ))
                    logZ_arr[k, idx] = logZ
                    score_arr[k, idx] = score

            # save the data
            out_dir = DIR_OUTPUT + os.sep + 'logZ' + os.sep + 'hopfield'
            fpath = out_dir + os.sep + 'objective_%dpatterns_%dsteps' % (k_patterns, steps)
            np.savez(fpath,
                     beta=beta_list,
                     termA=termA_arr,
                     logZ=logZ_arr,
                     score=score_arr)

            var_name = r'$\beta$'
            columns = [beta for beta in beta_list]

            score_name = r'$\langle\ln \ p(x)\rangle$'
            df_score = pd.DataFrame(score_arr, columns=columns).\
                melt(var_name=var_name, value_name=score_name)

            termA_name = r'$- \beta \langle H(s) \rangle$'
            df_termA = pd.DataFrame(termA_arr, columns=columns).\
                melt(var_name=var_name, value_name=termA_name)

            LogZ_name = r'$\ln \ Z$'
            df_LogZ = pd.DataFrame(logZ_arr, columns=columns).\
                melt(var_name=var_name, value_name=LogZ_name)

            plt.figure(); ax = sns.lineplot(x=var_name, y=score_name, data=df_score)
            plt.savefig(out_dir + os.sep + 'score_%dpatterns_%dsteps.pdf' % (k_patterns, steps)); plt.close()

            plt.figure(); ax = sns.lineplot(x=var_name, y=termA_name, data=df_termA)
            plt.savefig(out_dir + os.sep + 'termA_%dpatterns_%dsteps.pdf' % (k_patterns, steps)); plt.close()

            plt.figure(); ax = sns.lineplot(x=var_name, y=LogZ_name, data=df_LogZ)
            plt.savefig(out_dir + os.sep + 'LogZ_%dpatterns_%dsteps.pdf' % (k_patterns, steps)); plt.close()


    if specific_check:
        import time
        import torch

        beta = 2
        epoch_idx = [20]  #[0, 1, 99] #[96, 97, 98]
        AIS_STEPS = 2000
        nchains = 10
        # 96: Term A: 1417.3126916185463 | Log Z: 1419.820068359375 | Score: -2.507376740828704
        # 97: Term A: 1419.55814776832 | Log Z: 1443.4364013671875 | Score: -23.87825359886756
        # 98: Term A: 1419.647444170504 | Log Z: 1423.908203125 | Score: -4.260758954495941

        # (idx: 96) Term A: 1417.3126916185463 | Log Z: 1433.2881 | Score: -15.975394318953704
        # (idx: 97) Term A: 1419.55814776832 | Log Z: 1423.771 | Score: -4.212848325430059
        # (idx: 98) Term A: 1419.647444170504 | Log Z: 1432.9882 | Score: -13.340715009183441

        # (idx: 96) Term A: 1417.3126916185463 | Log Z: 1436.6534 | Score: -19.340750764266204
        # (idx: 97) Term A: 1419.55814776832 | Log Z: 1425.2229 | Score: -5.664752622305059
        # (idx: 98) Term A: 1419.647444170504 | Log Z: 1427.4364 | Score: -7.788957196683441

        # cv 1000 AIS: (don;t want to pass upper bound of -11 score (p = 1/60,000 for all sample points, implies 0 else)
        # (idx: 96) Term A: 1417.3126916185463 | Log Z: 1491.6993 | Score: -74.3866492017662
        # (idx: 97) Term A: 1419.55814776832 | Log Z: 1486.9297 | Score: -67.37153973168006
        # (idx: 98) Term A: 1419.647444170504 | Log Z: 1486.3145 | Score: -66.66700895449594

        # (idx: 96) Term A: 1417.3126916185463 | Log Z: 1493.4022 | Score: -76.0895300611412
        # (idx: 97) Term A: 1419.55814776832 | Log Z: 1493.8434 | Score: -74.28523602074256
        # (idx: 98) Term A: 1419.647444170504 | Log Z: 1484.3536 | Score: -64.70619352480844

        # load segment
        init_name = 'hopfield'  # hopfield or normal
        num_hid = 10
        run = 0
        bigruns = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs' + os.sep + 'rbm'
        subdir = '%s_%dhidden_0fields_2.00beta_100batch_100epochs_20cdk_1.00E-04eta_200ais' \
                 % (init_name, num_hid) + os.sep + 'run%d' % (run)
        fname = 'weights_%dhidden_0fields_20cdk_0stepsAIS_2.00beta.npz' % num_hid
        dataobj = np.load(bigruns + os.sep + subdir + os.sep + fname)
        weights_timeseries = dataobj['weights']

        X, _ = get_X_y_dataset(TRAINING, dim_visible=28**2, binarize=True)

        for idx in epoch_idx:
            weights = weights_timeseries[:, :, idx]
            rbm = weights_timeseries[:, :, idx]
            rbm = RBM_gaussian_custom(28**2, num_hid, 0, init_weights=None, use_fields=False, learning_rate=0)
            rbm.weights = torch.from_numpy(weights).float()

            print('Estimating term A...', )
            logP_termA = get_obj_term_A(X, rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta)

            """
            print('Estimating log Z (AIS)...', )
            logP_termB_forward, chains_state = esimate_logZ_with_AIS(rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta, num_steps=AIS_STEPS, num_chains=nchains)
            print('\tlogP_termB_forward:', logP_termB_forward)

            print('Estimating log Z (reverse AIS)...', )
            logP_termB_reverse, _ = esimate_logZ_with_reverse_AIS_algo2(rbm, rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta, num_steps=AIS_STEPS, num_chains=nchains)
            print('\tlogP_termB_reverse:', logP_termB_reverse)
            
            print('Estimating log Z (homemade AIS)...', )
            logP_termB_manual, chains_state = manual_AIS(rbm, beta, nchains=nchains, nsteps=AIS_STEPS)
            print('\tlogP_termB_manual:', logP_termB_manual)
            """

            print('Estimating log Z (homemade AIS reverse)...', )
            # Note settings from RAISE: 50 chains, 100,000 steps, 100 test cases. Think use control-variates thing as well (not implemented).
            # TODO implement control variates possibly
            rev_ntest = 10
            rev_nchain = 10
            rev_nsteps = AIS_STEPS
            test_cases = subsample_test_cases(X, rev_ntest)
            logP_termB_manual_reverse, _ = manual_AIS_reverse(rbm, beta, test_cases, nchains=rev_nchain, nsteps=rev_nsteps)
            print('\tlogP_termB_manual_reverse:', logP_termB_manual_reverse)
            """
            print('Estimating log Z (homemade AIS reverse Algo 3)...', )
            # Note settings from RAISE: 50 chains, 100,000 steps, 100 test cases. Think use control-variates thing as well (not implemented).
            # TODO implement control variates possibly
            rev_ntest = 10
            rev_nchain = 10
            rev_nsteps = AIS_STEPS
            test_cases = subsample_test_cases(X, rev_ntest)
            logP_termB_manual_reverse3, _ = manual_AIS_reverse(rbm, beta, test_cases, nchains=rev_nchain, nsteps=rev_nsteps)
            print('\tlogP_termB_manual_reverse3:', logP_termB_manual_reverse3)
            """

            #print('(idx: %d) AIS - Term A:' % idx, logP_termA, '| Log Z:', logP_termB_forward, '| Score:', logP_termA - logP_termB_forward)
            #print('(idx: %d) AIS (Reverse) - Term A:' % idx, logP_termA, '| Log Z:', logP_termB_reverse, '| Score:', logP_termA - logP_termB_reverse)
            #print('(idx: %d) Manual AIS - Term A:' % idx, logP_termA, '| Log Z:', logP_termB_manual, '| Score:', logP_termA - logP_termB_manual)
            #print('(idx: %d) Manual AIS (Reverse) - Term A:' % idx, logP_termA, '| Log Z:', logP_termB_manual_reverse, '| Score:', logP_termA - logP_termB_manual_reverse)
            print('(idx: %d) Manual AIS (Reverse 3) - Term A:' % idx, logP_termA, '| Log Z:', logP_termB_manual_reverse3, '| Score:', logP_termA - logP_termB_manual_reverse3)


            #chains_numpy = chains_state
            #chain_state_to_images(chains_numpy, rbm, 5, beta=beta)

    if compare_methods:
        import torch

        # 0) run parameters
        step_list = [10,50,100,500,1000]
        nchains = 100
        beta = 2.0

        logz_estimate_tf = np.zeros(len(step_list))
        logz_estimate_manual = np.zeros(len(step_list))
        logz_estimate_manual_rev = np.zeros(len(step_list))

        # 1) load an RBM
        init_name = 'hopfield'  # hopfield or normal
        epoch = 20
        num_hid = 10
        run = 0
        bigruns = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs' + os.sep + 'rbm'
        subdir = '%s_%dhidden_0fields_2.00beta_100batch_100epochs_20cdk_1.00E-04eta_200ais' \
                 % (init_name, num_hid) + os.sep + 'run%d' % (run)
        fname = 'weights_%dhidden_0fields_20cdk_0stepsAIS_2.00beta.npz' % num_hid
        dataobj = np.load(bigruns + os.sep + subdir + os.sep + fname)
        weights_timeseries = dataobj['weights']

        X, _ = get_X_y_dataset(TRAINING, dim_visible=28**2, binarize=True)

        weights = weights_timeseries[:, :, epoch]
        rbm = weights_timeseries[:, :, epoch]
        rbm = RBM_gaussian_custom(28 ** 2, num_hid, 0, init_weights=None, use_fields=False, learning_rate=0)
        rbm.weights = torch.from_numpy(weights).float()

        # 2) estimate "term A"
        print('Estimating term A...', )
        logP_termA = get_obj_term_A(X, rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta)

        for idx, steps in enumerate(step_list):

            # 2) perform TF AIS estimation
            print('Estimating log Z (AIS)...', )
            logP_termB_forward, _ = esimate_logZ_with_AIS(rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta, num_steps=steps, num_chains=nchains)
            logz_estimate_tf[idx] = logP_termB_forward

            # 3) perform custom AIS estimation
            print('Estimating log Z (homemade AIS)...', )
            logP_termB_manual, _ = manual_AIS(rbm, beta, nchains=nchains, nsteps=steps)
            logz_estimate_manual[idx] = logP_termB_manual

            # 4) perform custom AIS estimation
            #print('Estimating log Z (homemade AIS - reverse algo 2)...', )
            #logP_termB_manual_rev, _ = manual_AIS_reverse(rbm, beta, nchains=nchains, nsteps=steps)
            #logz_estimate_manual_rev[idx] = logP_termB_manual_rev

        # plots
        plt.plot(step_list, logz_estimate_tf, label='TF')
        plt.plot(step_list, logz_estimate_manual, label='manual')
        #plt.plot(step_list, logz_estimate_manual_rev, label='manual Reverse')
        plt.title('log Z')
        plt.legend()
        plt.show(); plt.close()

        plt.plot(step_list, logP_termA - logz_estimate_tf, label='TF')
        plt.plot(step_list, logP_termA - logz_estimate_manual, label='manual')
        #plt.plot(step_list, logP_termA - logz_estimate_manual_rev, label='manual Reverse')
        plt.title('score')
        plt.legend()
        plt.show(); plt.close()

        plt.plot(step_list, logz_estimate_tf, label='TF')
        plt.plot(step_list, logz_estimate_manual, label='manual')
        #plt.plot(step_list, logz_estimate_manual_rev, label='manual Reverse')
        plt.title('log Z')
        plt.xscale('log')
        plt.legend()
        plt.show(); plt.close()

        plt.plot(step_list, logP_termA - logz_estimate_tf, label='TF')
        plt.plot(step_list, logP_termA - logz_estimate_manual, label='manual')
        #plt.plot(step_list, logP_termA - logz_estimate_manual_rev, label='manual Reverse')
        plt.title('score')
        plt.xscale('log')
        plt.legend()
        plt.show(); plt.close()


        # save
        outdir = DIR_OUTPUT + os.sep + 'AIS'
        np.savetxt(outdir + os.sep + 'logz_estimate_tf.txt', logz_estimate_tf)
        np.savetxt(outdir + os.sep + 'logz_estimate_manual.txt', logz_estimate_manual)
        #np.savetxt(outdir + os.sep + 'logz_estimate_manual_rev.txt', logz_estimate_manual_rev)
        np.savetxt(outdir + os.sep + 'step_list.txt', step_list)
