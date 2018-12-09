import numpy as np
import matplotlib.pyplot as plt

"""
Example: langevin dynamics as perturbation to (deterministic) linear systems of differential equations

The ODE: autonomous 2x2 linear system
    - xvec_dot = A * x 
    - where A = [[-1,0], [0, -1]]
    
deterministic_dynamics(...) and langevin_dynamics(...) return output of the form:
    - states: np.array of size num_times x 2  (2D coordinates for this example)
    - times: np.array of size num_times
    - note on num_times:
        - in deterministic case: pre-determined (for euler method used here)
        - in langevin case case: pre-determined    
"""

STATE_DIM = 2
SYSTEM = np.array([[-1, 4],
                   [-0.5, -1]])


def deterministic_term(states, step, system=SYSTEM):
    """
    Calculating right hand side of vector equation: xdot = A .* x
    - system: the matrix A as an array
    """
    current_state = states[step, :]
    deterministic_term = np.dot(system, current_state)
    return np.dot(system, current_state)


def noise_term(dt):
    """
    Computed as part of Euler-Maruyama method for langevin dynamics
    noise_term looks like B(x_k)*delta_w
        - B(x_k) describes possibly state dependent, possibly anisotropic diffusion
        - delta_w = Norm(0, sqrt(dt))
    NOTE: we assume B(x_k) is a constant (i.e. scaled identity matrix)
    """
    delta_w = np.random.normal(0, np.sqrt(dt))
    return delta_w


def deterministic_dynamics(init_cond, dt, num_steps, init_time=0.0):
    """
    Uses naive euler's method: x(t+dt) = x_k + F(x_k, t_k) * dt
    """
    # prep arrays
    states = np.zeros((num_steps, STATE_DIM))
    times = np.zeros(num_steps)
    # fill init cond
    states[0, :] = init_cond
    times[0] = init_time
    for step in xrange(1, num_steps):
        states[step, :] = states[step-1, :] + deterministic_term(states, step-1) * dt
        times[step] = times[step-1] + dt
    return states, times


def langevin_dynamics(init_cond, dt, num_steps, init_time=0.0):
    """
    Uses Euler-Maruyama method: x(t+dt) = x_k + F(x_k, t_k) * dt + noise_term
    noise_term looks like B(x_k)*delta_w
        - B(x_k) describes possibly state dependent, possibly anisotropic diffusion
        - delta_w = Norm(0, sqrt(dt))
    """
    # prep arrays
    states = np.zeros((num_steps, STATE_DIM))
    times = np.zeros(num_steps)
    # fill init cond
    states[0, :] = init_cond
    times[0] = init_time
    for step in xrange(1, num_steps):
        states[step, :] = states[step-1, :] + deterministic_term(states, step-1) * dt + noise_term(dt)
        times[step] = times[step-1] + dt
    return states, times


if __name__ == '__main__':

    # settings
    init_cond = [10.0, 20.0]
    init_time = 4.0
    num_steps = 200
    dt = 0.1

    # get deterministic trajectory
    states, times = deterministic_dynamics(init_cond, dt, num_steps, init_time=init_time)

    # get langevin trajectories
    num_trials = 3
    trials_states = [0] * num_trials  # TODO array convert
    trials_times = [0] * num_trials  # TODO array convert
    for traj in xrange(num_trials):
        langevin_states, langevin_times = langevin_dynamics(init_cond, dt, num_steps, init_time=init_time)
        trials_states[traj] = langevin_states
        trials_times[traj] = langevin_times

    # plotting
    fig = plt.figure(figsize=(8, 6))
    plt.suptitle('Example comparison of deterministic vs langevin trajectories for 2D linear system')
    for subplt in xrange(2):
        ax = fig.add_subplot(1, 2, subplt + 1)
        ax.plot(times, states[:, subplt], label='deterministic')
        for traj in xrange(3):
            ax.plot(trials_times[traj], trials_states[traj][:, subplt], '--', alpha=0.4, label='stoch_%d' % traj)
        ax.legend()
        ax.set_xlabel('time')
        ax.set_ylabel('x%d' % subplt)
        plt.subplots_adjust(wspace=0.2)
    plt.show()
