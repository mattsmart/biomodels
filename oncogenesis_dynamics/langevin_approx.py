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
        noiseterm = np.dot(stoch_matrix, propensities_sqrt * randns) * dtsqrt
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
        'N': 10000.0,  # 100.0
        'v_x': 0.0,
        'v_y': 0.0,
        'v_z': 0.0,
        'mu_base': 0.0,
        'c2': 0.0,
        'v_z2': 0.0,
        'mult_inc': 1.0,
        'mult_dec': 1.0,
    }
    params = Params(params_dict, system, feedback=feedback)

    # main settings
    plot = True
    num_trials = 3
    num_to_plot = 3

    # trajectory settings
    init_time = 0.0
    num_steps = 200000
    dt = 1e-4
    init_cond = [params.N, 0, 0]

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
            ax.plot(times, states[:, state_idx], label='deterministic')
            #ax.axhline(steadystates[state_idx, 0], ls='--', c='k', alpha=0.4, label='FP formula')
            for traj in xrange(num_to_plot):
                ax.plot(trials_times[:, traj], trials_states[:, state_idx, traj], '--', alpha=0.4,
                        label='stoch_%d' % traj)
            ax.legend()
            ax.set_xlabel('time')
            ax.set_ylabel('%s' % params.states[state_idx])
            ax.set_title('State: %s' % params.states[state_idx])
            plt.subplots_adjust(wspace=0.2)
        plt.show()
