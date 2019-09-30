import numpy as np
import matplotlib.pyplot as plt

from params import Params
from trajectory import trajectory_simulate


# misc
FOLDER_OUTPUT = "output"
TIMESTEP=1e-3
NUM_STEPS=200

def langevin_dynamics(init_cond, params, dt=TIMESTEP, num_steps=NUM_STEPS, init_time=0.0, noise=1.0):
    """
    Uses Euler-Maruyama method: x(t+dt) = x_k + F(x_k, t_k) * dt + noise_term
    noise_term looks like B(x_k)*delta_w
        - B(x_k) describes possibly state dependent, possibly anisotropic diffusion
        - delta_w = Norm(0, sqrt(dt))
    Note: setting noise to 0.0 recovers deterministic dynamics
    Perturb slightly shifts the initial condition proportional to the noise

    Return output of the form:
    - states: np.array of size num_times x STATE_DIM
    - times: np.array of size num_times
    - note on num_times: size pre-determined (for Euler-Maruyama method used here)
    """
    # prep arrays
    states = np.zeros((num_steps, params.numstates))
    times = np.zeros(num_steps)
    # fill init cond
    states[0, :] = init_cond
    times[0] = init_time
    # misc
    dtsqrt = np.sqrt(dt)
    stoch_matrix = np.array([[1, -1, 0,  0, 0,  0, -1,  1,  0],
                             [0,  0, 1, -1, 0,  0,  1, -1, -1],
                             [0,  0, 0,  0, 1, -1,  0,  0,  1]])

    for step in xrange(1, num_steps):
        current_state = states[step-1, :]
        #print current_state, step
        propensities = np.array(params.rxn_prop(current_state))[0:9]  # note could append fpt event as in formulae.py
        propensities_sqrt = np.sqrt(propensities)

        # deterministic term
        determ = np.dot(stoch_matrix, propensities) * dt
        # noise term
        # .. .sample normals len prop
        randns = np.random.normal(0, 1.0, len(propensities))
        # .. use propensities_sqrt
        noiseterm = dtsqrt * np.dot(np.dot(stoch_matrix, np.diag(propensities_sqrt)), randns)
        # update
        next_state = current_state + determ + noiseterm
        states[step, :] = np.maximum(next_state, np.zeros(3))

        times[step] = times[step - 1] + dt

    return states, times


def gen_multitraj(num_trials, init_cond, params, dt=TIMESTEP, num_steps=NUM_STEPS, init_time=0.0):
    # TODO try init cond as FP
    trials_states = np.zeros((num_steps, params.numstates, num_trials))
    trials_times = np.zeros((num_steps, num_trials))
    # setup init cond
    for traj in xrange(num_trials):
        print traj
        langevin_states, langevin_times = langevin_dynamics(init_cond, params, dt=dt, num_steps=num_steps,
                                                            init_time=init_time)
        trials_states[:, :, traj] = langevin_states
        trials_times[:, traj] = langevin_times
    return trials_states, trials_times


def get_FPE_LNA_gaussian_diff(params):
    dims = 3
    assert params.mult_inc == 100
    xFP = np.array([77.48756569595079, 22.471588735222426, 0.04084556882678214]) / 100.0 * params.N
    xSaddle = np.array([40.61475564788107, 40.401927055159106, 18.983317296959825]) / 100.0 * params.N

    # J from mathematica
    J_true = np.array([[-0.0240001, 1.4124, -0.697388],
                       [-0.930607, -2.36718, -0.202244],
                       [-0.000408456, -0.000226765, -0.0553836]])
    # D direct compute
    stoch_matrix = np.array([[1, -1, 0,  0, 0,  0, -1,  1,  0],
                             [0,  0, 1, -1, 0,  0,  1, -1, -1],
                             [0,  0, 0,  0, 1, -1,  0,  0,  1]])
    propensities = np.array(params.rxn_prop(xFP))[0:9]  # note could append fpt event as in formulae.py
    D_true = np.dot(np.dot(stoch_matrix, np.diag(propensities)), stoch_matrix.T)

    import scipy as sp
    cov = sp.linalg.solve_lyapunov(J_true, -D_true)  # this is wrong
    print propensities
    print D_true
    print cov
    print 'linalg'
    print np.linalg.eig(J_true)
    covInv = np.linalg.inv(cov)
    covSqrtDet = np.sqrt(np.linalg.det(cov))
    def sol(x):
        A = ((2 * np.pi) ** (0.5 * dims) * covSqrtDet) ** -1
        dev = x - xFP
        B = np.exp(-0.5 * np.dot(np.dot(dev.T, covInv), dev))
        print A,B,dev, np.dot(np.dot(dev.T, covInv), dev)
        return A*B
    high = sol(xFP)
    low = sol(xSaddle)
    print high, low, high - low
    return high - low



if __name__ == '__main__':

    # DYNAMICS PARAMETERS
    system = "feedback_z"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
    feedback = "tanh"  # "constant", "hill", "step", "pwlinear"
    params_dict = {
        'alpha_plus': 0.2,
        'alpha_minus': 1.0,  # 0.5
        'mu': 1e-4,  # 0.01
        'a': 1.0,
        'b': 1.2,
        'c': 1.1,  # 1.2
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

    # main settings
    plot = True
    num_trials = 10
    num_to_plot = 10

    # trajectory settings
    init_time = 0.0
    num_steps = 2000*10
    dt = 1e-3
    init_cond = [params.N, 0, 0]

    get_FPE_LNA_gaussian_diff(params)

    # get deterministic trajectory
    states, times = trajectory_simulate(params, init_cond=init_cond, t0=0.0, t1=num_steps*dt, num_steps=num_steps,
                                        sim_method="libcall")

    # get langevin trajectories
    trials_states, trials_times = gen_multitraj(num_trials, init_cond, params, dt=dt, num_steps=num_steps,
                                                init_time=0.0)

    # plotting master genes
    if plot:
        fig = plt.figure(figsize=(8, 6))
        plt.suptitle('Comparison of deterministic vs langevin xyz sim')
        for state_idx in xrange(3):
            ax = fig.add_subplot(1, 3, state_idx + 1)
            ax.plot(times, states[:, state_idx], 'k', label='deterministic')
            #ax.axhline(steadystates[state_idx, 0], ls='--', c='k', alpha=0.4, label='FP formula')
            for traj in xrange(num_to_plot):
                ax.plot(trials_times[:, traj], trials_states[:, state_idx, traj], '--', alpha=0.3,
                        label='stoch_%d' % traj)
            # plot mean of trials
            ax.plot(np.mean(trials_times[:, :], axis=1), np.mean(trials_states[:, state_idx, :], axis=1), '--k',
                    alpha=0.8, label='stoch_mean')
            #ax.legend()
            ax.set_xlabel('time')
            ax.set_ylabel('%s' % params.states[state_idx])
            ax.set_title('State: %s' % params.states[state_idx])
            plt.subplots_adjust(wspace=0.2)
        plt.show()
