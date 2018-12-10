import numpy as np
import matplotlib.pyplot as plt

from settings import DEFAULT_PARAMS, FOLDER_OUTPUT, STATE_SLAVE_DIM

"""
Encode gene expression dynamics described in July pdf
    - form of the dynamics is assumed
    - as model parameter is vaired there is a patchfork bifurcation
    - assume linearized dynamics near fixed point
    
The ODE: autonomous STATE_DIM x STATE_DIM linear system
    - xdot = J * (x - x_steadystate)
    - the jacobian J is defined by model parameters, a control parameter, and the steady state
    
Model assumptions:
    - there are 2 master genes which interact with eachother and may control expression of other slave genes
    - the slave genes do not interact with other genes
    - here assume the first master gene (i.e. index 0) controls other genes

Dynamics form:
    - x_dot = 1/(1 + y^2) - x/tau
    - y_dot = (1/(1 + x^2) - y/tau) * gamma
    - v_i_dot = beta_i * ((alpha_i*x^2 + 1 - alpha_i)/(1+x^2)) - v_i/tau_i

Model parameters (for master genes: x, y):
    - (*** UNUSED CURRENTLY ***) h: hill parameter; nonlinearity of gene interaction, for now we set it to 1
    - tau: (equivalent) degradation rate of each master gene
    - gamma: sets timescale for y_dot dynamics (scales whole RHS)

Model parameters (for slave genes: v_i):
    - alpha_i: 0<=alpha<=1, one for each slave gene, controls activation (1) or repression (0) by a master gene
    - beta_i: scales the production term for a slave gene
    - tau_i: exponential degradation rate  
"""


def jacobian_pitchfork(params, steadystate):
    # aliases
    p = params
    xss = steadystate[0]
    yss = steadystate[1]
    # check + jac prep
    assert p.dim_master == 2  # TODO generalize this
    jac = np.zeros((p.dim, p.dim))
    # specify master gene components of J_ij
    jac[0, 0] = -1.0 / p.tau
    jac[0, 1] = -2.0 * yss / (1 + yss**2)
    jac[1, 0] = -p.gamma * 2.0 * xss / (1 + xss**2)
    jac[1, 1] = -p.gamma / p.tau
    # specify slave gene components of J_ij
    for i in xrange(2, p.dim):
        slave_idx = i - p.dim_master
        jac[i, 0] = p.betas[slave_idx] * (2 * xss) * (2*p.alphas[slave_idx] - 1) / ((xss**2 + 1) ** 2)
        jac[i, i] = -1.0 / p.taus[slave_idx]
    return jac


def steadystate_pitchfork(params):
    # TODO note may need to pass output to jacobian call
    p = params
    assert p.dim_master == 2  # TODO generalize this
    steadystate = np.zeros(p.dim)
    # master steady states
    # TODO
    # slaves steady states
    # TODO
    for i in xrange(2, p.dim):
        slave_idx = i - p.dim_master
        steadystate[i] = None
    return 0


def deterministic_term(states, step, jacobian, steady_state):
    """
    Calculating right hand side of vector equation: xdot = J * (x - x_steadystate)
    """
    current_state = states[step, :]
    state_difference = current_state - steady_state
    return np.dot(jacobian, state_difference)


def noise_term(dt):
    # TODO more generally involves matrix product to get N x 1 term: B*dW is N x k * k x 1 where N = STATE_DIM
    # TODO noise should be diagonal with something like sqrt(2 <x_i> / tau_i), check orig script for form
    """
    Computed as part of Euler-Maruyama method for langevin dynamics
    noise_term looks like B(x_k)*delta_w
        - B(x_k) describes possibly state dependent, possibly anisotropic diffusion
        - delta_w = Norm(0, sqrt(dt))
    NOTE: we assume B(x_k) is a constant (i.e. scaled identity matrix)
    """
    delta_w = np.random.normal(0, np.sqrt(dt))
    return delta_w


def langevin_dynamics(init_cond, dt, num_steps, init_time=0.0, params=DEFAULT_PARAMS, noise=1.0):
    """
    Uses Euler-Maruyama method: x(t+dt) = x_k + F(x_k, t_k) * dt + noise_term
    noise_term looks like B(x_k)*delta_w
        - B(x_k) describes possibly state dependent, possibly anisotropic diffusion
        - delta_w = Norm(0, sqrt(dt))
    Note: setting noise to 0.0 recovers deterministic dynamics

    return output of the form:
    - states: np.array of size num_times x STATE_DIM
    - times: np.array of size num_times
    - note on num_times: size pre-determined (for Euler-Maruyama method used here)
    """
    # prep arrays
    states = np.zeros((num_steps, params.dim))
    times = np.zeros(num_steps)
    # fill init cond
    states[0, :] = init_cond
    times[0] = init_time
    # build model
    jacobian = jacobian_pitchfork(params)
    steadystate = steadystate_pitchfork(params)
    for step in xrange(1, num_steps):
        states[step, :] = states[step-1, :] + noise * noise_term(dt) + \
                          deterministic_term(states, step-1, jacobian, steadystate) * dt
        times[step] = times[step-1] + dt
    return states, times


if __name__ == '__main__':

    # setup params
    params = DEFAULT_PARAMS
    print(params)

    # trajectory settings
    init_cond = [10.0, 25.0] + [0 for _ in xrange(params.dim_slave)]
    init_time = 4.0
    num_steps = 200
    dt = 0.1

    # get deterministic trajectory
    states, times = langevin_dynamics(init_cond, dt, num_steps, init_time=init_time, params=params, noise=0.0)

    # get langevin trajectories
    num_trials = 3
    trials_states = [0] * num_trials  # TODO array convert
    trials_times = [0] * num_trials  # TODO array convert
    for traj in xrange(num_trials):
        langevin_states, langevin_times = langevin_dynamics(init_cond, dt, num_steps, init_time=init_time, params=params)
        trials_states[traj] = langevin_states
        trials_times[traj] = langevin_times

    # plotting master genes
    fig = plt.figure(figsize=(8, 6))
    plt.suptitle('Comparison of deterministic vs langevin gene expression for pitchfork system')
    for state_idx in xrange(2):
        ax = fig.add_subplot(1, 2, state_idx + 1)
        ax.plot(times, states[:, state_idx], label='deterministic')
        for traj in xrange(num_trials):
            ax.plot(trials_times[traj], trials_states[traj][:, state_idx], '--', alpha=0.4, label='stoch_%d' % traj)
        ax.legend()
        ax.set_xlabel('time')
        ax.set_ylabel('x%d' % state_idx)
        plt.subplots_adjust(wspace=0.2)
    plt.show()
