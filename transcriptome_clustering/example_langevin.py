import numpy as np
import matplotlib.pyplot as plt

"""
Example: langevin dynamics as perturbation to (deterministic) linear systems of differential equations

The ODE: autonomous 2x2 linear system
    - xvec_dot = A * x 
    - where A = [[-1,0], [0, -1]]
"""


def deterministic_dynamics():
    # TODO
    return states, times


def langevin_dynamics():
    # TODO
    return states, times


if __name__ == '__main__':
    states, times = deterministic_dynamics()

    num_trials = 3
    trials_states = [0] * num_trials  # TODO array convert
    trials_times = [0] * num_trials  # TODO array convert
    for traj in xrange(3):
        langevin_states, langevin_times = langevin_dynamics()
        trials_states[traj] = langevin_states
        trials_times[traj] = langevin_times

    fig = plt.figure(figsize=(8, 6))
    plt.suptitle('Example comparison of deterministic vs langevin trajectories for 2D linear system')
    for subplt in xrange(2):
        ax = fig.add_subplot(1, 2, subplt + 1)
        ax.plot(times, states[:, subplt], label='deterministic')
        for traj in xrange(3):
            ax.plot(trials_times[traj], trials_states[traj][:, subplt], label='langevin_traj_%d' % traj)
        ax.legend()
        ax.xlabel('time')
        ax.ylabel('x%d' % subplt)
