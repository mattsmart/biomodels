import numpy as np
import matplotlib.pyplot as plt

from settings import DEFAULT_PARAMS, FOLDER_OUTPUT, STATE_SLAVE_DIM
from statistical_formulae import build_diffusion, build_covariance, infer_interactions

"""
TODO
1 - can have negative master/slave states, for both linearized and direct dynamics -- FIX
  - this happens even without noise
2 - fluctuation can send state below zero (separate problem from 1)
"""

"""
Encode gene expression dynamics described in July pdf
    - form of the dynamics is assumed
    - as model parameter is vaired there is a patchfork bifurcation
    - assume linearized dynamics near fixed point
    
The ODE: autonomous STATE_DIM x STATE_DIM linear system
    - xdot = J * (x - x_steadystate)
    - the jacobian J is defined by model parameters, a control parameter, and the steady state
    
General procedure:
- model starts before bifurcation, and as a key param is tuned a bifurcation occurs
- fix model parameters   -> run N trajectories -> compute steady state correlation function -> infer J_ij
- slide model parameters -> run N trajectories -> compute steady state correlation function -> infer J_ij 
- ...
- report bifurcation point if/when eig(J_ij) -> 0
- at the detected bifurcation, report on control genes via deletion method
    
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
    """
    Assumes steady state is an array of size state_dim, but only uses the first two components (xss, yss)
    Returns: state_dim x state_dim array
    """
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
    """
    See mathematica file -- there is 1 real root if 0 < tau <= 2, 3 for tau > 2 (the 3 are coincident at tau=2)
    Returns steady states array of form: state_dim x num_fixed_points
        - should be DIM x 1 or DIM x 3 typically
    """
    # TODO note may need to pass output to jacobian call
    def yss_root_main(tau):
        C = (9*tau + np.sqrt(12 + 81 * tau ** 2))**(1.0/3.0)
        num = -2 * 3**(1.0/3.0) + 2**(1.0/3.0) * C ** 2
        den = 6**(2.0/3.0) * C
        return num / den

    def yss_root_plus(tau):
        return 0.5 * (tau + np.sqrt(tau ** 2 - 4))

    def yss_root_minus(tau):
        return 0.5 * (tau - np.sqrt(tau ** 2 - 4))

    def xss_from_yss(yss, tau):
        return tau / (1 + yss**2)

    def vss_from_xss(xss, alpha_i, beta_i, tau_i):
        return tau_i * beta_i * (alpha_i * xss**2 + 1 - alpha_i) / (1 + xss**2)

    root_to_int = {0: yss_root_main, 1: yss_root_plus, 2: yss_root_minus}

    p = params
    assert p.dim_master == 2

    # two main cases for master genes
    if 0 < p.tau <= 2:
        num_fp = 1
        steadystates = np.zeros((p.dim, num_fp))
        steadystates[1][0] = yss_root_main(p.tau)
        steadystates[0][0] = xss_from_yss(steadystates[1][0], p.tau)
    else:
        num_fp = 3
        steadystates = np.zeros((p.dim, num_fp))
        for fp in xrange(num_fp):
            yss_fn = root_to_int[fp]
            steadystates[1][fp] = yss_fn(p.tau)
            steadystates[0][fp] = xss_from_yss(steadystates[1][fp], p.tau)
    # slaves steady states
    for fp in xrange(num_fp):
        xss = steadystates[0][fp]
        for i in xrange(2, p.dim):
            slave = i - p.dim_master
            steadystates[i][fp] = vss_from_xss(xss, p.alphas[slave], p.betas[slave], p.taus[slave])
    return steadystates


def deterministic_term(states, step, params, linearized=False, jacobian=None, fp=None):
    """
    Calculating right hand side of vector equation: xdot = F(x)
    If linearized: need to pass jacobian and a fixed point to compute linearized dynamics xdot=J*(x-x_fp)
    """
    p = params
    current_state = states[step, :]
    if linearized:
        state_difference = current_state - fp
        rhs = np.dot(jacobian, state_difference)
    else:
        rhs = np.zeros(p.dim)
        assert p.dim_master == 2
        x = current_state[0]
        y = current_state[1]
        rhs[0] = 1/(1 + y**2) - x/p.tau                     # x_dot RHS
        rhs[1] = (1/(1 + x**2) - y/p.tau) * p.gamma         # y_dot RHS
        for idx in xrange(2, p.dim):
            slave = idx - p.dim_master
            alpha, beta, tau = p.alphas[slave], p.betas[slave], p.taus[slave]
            rhs[idx] = beta * (alpha * x ** 2 + 1 - alpha) / (1 + x ** 2) - current_state[idx] / tau   # vi_dot RHS
    return rhs


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
    linearized = False
    if linearized:
        steadystates = steadystate_pitchfork(params)
        fp_mid = steadystates[:, 0]                                 # always linearize around middle branch FP
        jacobian = jacobian_pitchfork(params, fp_mid)

    for step in xrange(1, num_steps):
        if linearized:
            determ = deterministic_term(states, step - 1, params, linearized=True, jacobian=jacobian, fp=fp_mid)
        else:
            determ = deterministic_term(states, step - 1, params, linearized=False)
        states[step, :] = states[step-1, :] + noise * noise_term(dt) + determ * dt
        times[step] = times[step-1] + dt

    return states, times


if __name__ == '__main__':

    # setup params
    params = DEFAULT_PARAMS
    params.printer()

    # trajectory settings
    init_cond = [10.0, 25.0] + [0 for _ in xrange(params.dim_slave)]
    init_time = 4.0
    num_steps = 200
    dt = 0.1

    # get deterministic trajectory
    states, times = langevin_dynamics(init_cond, dt, num_steps, init_time=init_time, params=params, noise=0.0)

    # get langevin trajectories
    num_trials = 3
    trials_states = np.zeros((num_steps, params.dim, num_trials))
    trials_times = np.zeros((num_steps, num_trials))
    for traj in xrange(num_trials):
        langevin_states, langevin_times = langevin_dynamics(init_cond, dt, num_steps, init_time=init_time, params=params)
        trials_states[:, :, traj] = langevin_states
        trials_times[:, traj] = langevin_times

    # plotting master genes
    fig = plt.figure(figsize=(8, 6))
    plt.suptitle('Comparison of deterministic vs langevin gene expression for pitchfork system')
    for state_idx in xrange(2):
        ax = fig.add_subplot(1, 2, state_idx + 1)
        ax.plot(times, states[:, state_idx], label='deterministic')
        for traj in xrange(num_trials):
            ax.plot(trials_times[:, traj], trials_states[:, state_idx, traj], '--', alpha=0.4, label='stoch_%d' % traj)
        ax.legend()
        ax.set_xlabel('time')
        ax.set_ylabel('%s' % params.state_dict[state_idx])
        ax.set_title('State: %s' % params.state_dict[state_idx])
        plt.subplots_adjust(wspace=0.2)
    plt.show()

    # print diffusion, covariance, J_ij
    D = build_diffusion(trials_states, params)
    C = build_covariance(trials_states, params)
    J = infer_interactions(trials_states, params)
    print "D - diffusion"
    print D
    print "C - covariance"
    print C
    print "J - interactions"
    print J
