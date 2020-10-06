import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def run_tf_example_A():
    # EXAMPLE CODE
    # https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/sample_annealed_importance_chain
    tfd = tfp.distributions

    # Run 100 AIS chains in parallel
    num_chains = 100
    dims = 20
    dtype = np.float32

    proposal = tfd.MultivariateNormalDiag(
       loc=tf.zeros([dims], dtype=dtype))

    target = tfd.TransformedDistribution(
      distribution=tfd.Gamma(concentration=dtype(2),
                             rate=dtype(3)),
      bijector=tfp.bijectors.Invert(tfp.bijectors.Exp()),
      event_shape=[dims])

    chains_state, ais_weights, kernels_results = (
        tfp.mcmc.sample_annealed_importance_chain(
            num_steps=1000,
            proposal_log_prob_fn=proposal.log_prob,
            target_log_prob_fn=target.log_prob,
            current_state=proposal.sample(num_chains),
            make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
              target_log_prob_fn=tlp_fn,
              step_size=0.2,
              num_leapfrog_steps=2)))

    log_estimated_normalizer = (tf.reduce_logsumexp(ais_weights)
                                - np.log(num_chains))
    log_true_normalizer = tf.math.lgamma(2.) - 2. * tf.math.log(3.)

    print("True", log_true_normalizer)
    print("Estimated", log_estimated_normalizer)


def run_test_ising_3spin(beta=2.0, nsteps=10, nchains=100):
    # EXAMPLE CODE
    # N = 3 spins
    # Jij = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]  # should behave like ferromagnet
    # Z for this case is computable exactly

    # Run 100 AIS chains in parallel
    num_chains = nchains
    dims = 1  # use p-form continuous rep. for integral
    dims_N = 3
    dtype = tf.float32

    # fix target model
    #Jij = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])  # TODO add diagonals compare
    Jij = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])  # TODO add diagonals compare, will change ln Z by adding beta/2.0 * Tr(J)
    WEIGHTS = np.array([[1.0, 1.0, 1.0]]).T
    Jij_tf = tf.convert_to_tensor(Jij, dtype=dtype)
    WEIGHTS_tf = tf.convert_to_tensor(WEIGHTS, dtype=dtype)

    """
    def proposal_log_prob_fn(*states):
        # given vector size N ints, return scalar for each chain
        #fvals = [1.0] * len(states)
        fvals = tf.ones(len(states), dtype=tf.float32)
        return fvals

    def target_log_prob_fn(*states):
        # TODO 1: the state must be binary but appears as floats (when printed during the sim)
        #         maybe use Metropolis Hastings instead of HMC (requires continuous p(x))
        # TODO 2: if the state must be continuous, maybe we switch to the p-dim hidden variable form and treat the integrand as e^S(h) and use S(h) as our log-prob
        # given vector size N ints, return scalar for each chain
        fvals = [0.0] * len(states)
        for idx, state in enumerate(states):
            print(Jij_tf)
            print('BINARY?', state)
            #print(tf.tensordot(Jij_tf, state, 1))
            negative_energy = 0.5 * tf.tensordot(state, tf.tensordot(Jij_tf, state, 1), 1)
            print(negative_energy)
            fvals[idx] = beta * negative_energy
        fvals = tf.convert_to_tensor(fvals, dtype=tf.float32)
        return fvals
        
    init_state = [0] * num_chains
    for idx in range(num_chains):
        sample_01_convention = np.random.binomial(1, 0.5, 3)  # this should sample the uniform distribution on 3 spins
        sample = sample_01_convention * 2 - 1
        init_state[idx] = tf.convert_to_tensor(sample, dtype=dtype)
    """
    tfd = tfp.distributions

    proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([dims], dtype=dtype))
    proposal_log_prob_fn = proposal.log_prob


    target_log_prob_const = dims_N * tf.math.log( 2.0 ) - (dims / 2.0) * tf.math.log( 2.0 * np.pi / beta)
    print("target_log_prob_const", target_log_prob_const)

    def target_log_prob_fn(hidden_states):
        # TODO 1: the state must be binary but appears as floats (when printed during the sim)
        #         maybe use Metropolis Hastings instead of HMC (requires continuous p(x))
        # TODO 2: if the state must be continuous, maybe we switch to the p-dim hidden variable form and treat the integrand as e^S(h) and use S(h) as our log-prob
        # given vector size N ints, return scalar for each chain
        fvals = [0.0] * len(hidden_states)
        # TODO tensor speedup test with p > 1
        for idx, hidden in enumerate(hidden_states):
            term1 = tf.tensordot(hidden, hidden, 1)
            cosh_arg = beta * tf.tensordot(WEIGHTS_tf, hidden, 1)
            log_cosh_vec = tf.math.log( tf.math.cosh(cosh_arg) )
            term2 = tf.math.reduce_sum(log_cosh_vec)
            fvals[idx] = - (beta / 2.0) * term1 + term2
        fvals = tf.convert_to_tensor(fvals, dtype=dtype) + target_log_prob_const
        return fvals

    # draw 100 samples from the proposal distribution
    init_state = proposal.sample(num_chains)
    #print(type(init_state))
    #print(init_state)
    #print('.........')

    chains_state, ais_weights, kernels_results = (
        tfp.mcmc.sample_annealed_importance_chain(
            num_steps=nsteps,
            proposal_log_prob_fn=proposal_log_prob_fn,
            target_log_prob_fn=target_log_prob_fn,
            current_state=init_state,
            make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
              target_log_prob_fn=tlp_fn,
              step_size=0.2,
              num_leapfrog_steps=2)))

    log_estimated_normalizer = (tf.reduce_logsumexp(ais_weights)
                                - np.log(num_chains))

    # compute true analytically
    states = [np.array([-1, -1, -1]),
              np.array([-1, -1,  1]),
              np.array([-1,  1, -1]),
              np.array([-1,  1,  1]),
              np.array([ 1, -1, -1]),
              np.array([ 1, -1,  1]),
              np.array([ 1,  1, -1]),
              np.array([ 1,  1,  1])]
    beta = beta  # TODO care
    boltz_factors = [np.exp(0.5 * beta * np.dot(s.T, np.dot(Jij, s))) for s in states]
    Z = np.sum(boltz_factors)
    log_true_normalizer = np.log(Z)

    print("True", log_true_normalizer)
    print("Estimated", log_estimated_normalizer)
    return log_estimated_normalizer


if __name__ == '__main__':

    #print("Running example A...")
    #run_tf_example_A()

    print("Running example B...")
    #log_estimated_normalizer = run_test_ising_3spin()
    nn = 10
    runs = [0] * nn
    for idx in range(nn):
        runs[idx] = run_test_ising_3spin(beta=2.0)
    print(runs)
