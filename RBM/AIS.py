import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
#print(tf.config.threading.get_inter_op_parallelism_threads())
#print(tf.config.threading.get_intra_op_parallelism_threads())
#tf.config.threading.set_inter_op_parallelism_threads(12)
#tf.config.threading.set_intra_op_parallelism_threads(12)

import pandas as pd
import seaborn as sns; sns.set()

from RBM_train import load_rbm_hopfield, TRAINING
from RBM_assess import get_X_y_dataset
from settings import DIR_OUTPUT, DIR_MODELS


def esimate_logZ_with_AIS(weights, field_visible, field_hidden, beta=1.0, num_chains=100, num_steps=1000):
    # TODO have both fields -- how to incorporate
    # Note: doing 9 'runs' of this for 1000 steps was ~1.5hr
    # Run num_chains AIS chains in parallel for num_steps
    N = field_visible.shape[0]
    p = field_hidden.shape[0]
    assert weights.shape[0] == N
    assert weights.shape[1] == p

    dims = p
    dtype = tf.float32

    # fix target model  # TODO minor speedup if we move out
    weights_tf = tf.convert_to_tensor(weights, dtype=dtype)
    weights_tf = tf.convert_to_tensor(field_visible, dtype=dtype)
    weights_tf = tf.convert_to_tensor(field_hidden, dtype=dtype)

    # define proposal distribution
    tfd = tfp.distributions
    proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([dims], dtype=dtype))
    proposal_log_prob_fn = proposal.log_prob

    # define target distribution
    target_log_prob_const = N * np.log(2.0) - (dims / 2.0) * np.log(2.0 * np.pi / beta)
    print("target_log_prob_const", target_log_prob_const)

    def target_log_prob_fn(hidden_states):
        # given vector size N ints, return scalar for each chain
        fvals = [0.0] * len(hidden_states)
        # TODO tensor speedup test with p > 1
        for idx, hidden in enumerate(hidden_states):

            hidden_to_sqr = hidden - field_hidden
            term1 = tf.tensordot(hidden_to_sqr, hidden_to_sqr, 1)

            cosh_arg = beta * (tf.tensordot(weights_tf, hidden, 1) + field_visible)
            log_cosh_vec = tf.math.log(tf.math.cosh(cosh_arg))
            term2 = tf.math.reduce_sum(log_cosh_vec)

            fvals[idx] = - (beta / 2.0) * term1 + term2
        fvals = tf.convert_to_tensor(fvals, dtype=dtype) + target_log_prob_const
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

    return log_Z.numpy()


def get_obj_term_A(dataset_prepped, weights, field_visible, field_hidden, beta=1.0):
    data_indep_term = - (beta / 2.0) * np.dot(field_hidden, field_hidden)

    data_sum = 0
    for x in dataset_prepped:
        A = np.dot(field_visible, x)
        Bvec = field_hidden + np.dot(weights.T, x)
        data_sum += A + 0.5 * np.dot(Bvec, Bvec)

    val_avg = beta * data_sum / len(dataset_prepped) + data_indep_term
    return val_avg


if __name__ == '__main__':
    # AIS settings
    steps = 10  # 1000 and 5000 similar, very slow

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
        runs = 1
        beta_list = np.linspace(0.5, 20, 20).astype(np.float32)
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
                logZ = esimate_logZ_with_AIS(weights, visible_field, hidden_field, beta=beta, num_steps=steps)
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
