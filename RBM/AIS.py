import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import tensorflow as tf
import tensorflow_probability as tfp
#print(tf.config.threading.get_inter_op_parallelism_threads())
#print(tf.config.threading.get_intra_op_parallelism_threads())
#tf.config.threading.set_inter_op_parallelism_threads(12)
#tf.config.threading.set_intra_op_parallelism_threads(12)

import pandas as pd
import seaborn as sns; sns.set()

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
    return log_Z.numpy(), chains_state


def get_obj_term_A(dataset_prepped, weights, field_visible, field_hidden, beta=1.0):
    data_indep_term = - (beta / 2.0) * np.dot(field_hidden, field_hidden)

    data_sum = 0
    for x in dataset_prepped:
        A = np.dot(field_visible, x)
        Bvec = field_hidden + np.dot(weights.T, x)
        data_sum += A + 0.5 * np.dot(Bvec, Bvec)

    val_avg = beta * data_sum / len(dataset_prepped) + data_indep_term
    return val_avg


def chain_state_to_images(chain_state, rbm, num_images, beta=200.0):
    chains_numpy = chain_state.numpy()
    num_chains = chains_numpy.shape[0]
    print(chains_numpy.shape)

    for idx in range(num_images):
        hidden_state_numpy = chains_numpy[idx, :]
        hidden_state_torch = torch.from_numpy(hidden_state_numpy)
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
        import torch
        from custom_rbm import RBM_gaussian_custom

        beta = 2
        epoch_idx = [0, 1, 99]#[96, 97, 98]
        AIS_STEPS = 200
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

        bigruns = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs' + os.sep + 'rbm'
        subdir = 'E_extra_beta2duringTraining_100batch_100epochs_20cdk_1.00E-04eta_200ais' + os.sep + 'run1'
        fname = 'weights_50hidden_0fields_20cdk_200stepsAIS_2.00beta.npz'
        dataobj = np.load(bigruns + os.sep + subdir + os.sep + fname)
        weights_timeseries = dataobj['weights']

        X, _ = get_X_y_dataset(TRAINING, dim_visible=28**2, binarize=True)

        for idx in epoch_idx:
            weights = weights_timeseries[:, :, idx]
            rbm = weights_timeseries[:, :, idx]
            rbm = RBM_gaussian_custom(28**2, 50, 0, load_init_weights=False, use_fields=False, learning_rate=0)
            rbm.weights = torch.from_numpy(weights).float()

            print('Estimating term A...', )
            logP_termA = get_obj_term_A(X, rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta)
            print('Estimating log Z...', )
            logP_termB, chains_state = esimate_logZ_with_AIS(rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta, num_steps=AIS_STEPS)
            print('(idx: %d) Term A:' % idx, logP_termA, '| Log Z:', logP_termB, '| Score:', logP_termA - logP_termB)
            chain_state_to_images(chains_state, rbm, 2, beta=beta)
