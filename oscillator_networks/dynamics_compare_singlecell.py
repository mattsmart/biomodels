import matplotlib.pyplot as plt
import numpy as np
import os

from preset_solver import PRESET_SOLVER
from class_singlecell import SingleCell


def compare_solvers_singlecell(single_cell, solver_presets, timer_mode=False):
    """
    Given a list of solvers, compute SingleCell trajectories for each.

    Inputs:
        single_cell class object
        solvers_presets: list of dicts which match the format in "preset_solver.py"
        timer_mode: if True, run each traj 10 times and return mean/min/max of times
    Create a plot with N rows, 1 column. Each row has the following form:
        x_1: distinct curve corresponding to each solver
        ...
        x_N: distinct curve corresponding to each solver

    Return trajectories for each as arrays
    """
    if timer_mode:
        print('TODO implement')
        assert 1==2

    nrow = single_cell.dim_ode  # number of sc state variables

    nsolvers = len(solver_presets)
    traj_obj = []
    solver_times = [0] * nsolvers
    solver_traj = [0] * nsolvers

    for solver_idx in range(nsolvers):
        solver = solver_presets[solver_idx]
        solver_label = solver['label']
        solver_dynamics_method = solver['dynamics_method']  # e.g. 'solve_ivp'
        solver_kwargs = solver['kwargs']                    # e.g. 'dict(method='Radau')'

        print("Computing traj for solver: %s" % solver_label)
        print('Timer mode:', timer_mode)
        r, times = sc.trajectory(flag_info=True, dynamics_method=solver_dynamics_method, **solver_kwargs)
        solver_times[solver_idx] = times
        solver_traj[solver_idx] = r.T

    # plotting
    fig, axarr = plt.subplots(ncols=1, nrows=nrow, figsize=(8, 8), constrained_layout=True, squeeze=False, sharex=True)

    """
    state_tensor = self.state_to_rectangle(self.state_history)
    times = self.times_history
    birthdays = self.cell_stats[:, 2]
    """

    # Set plot labels and title
    y_axis_labels = [single_cell.variables_short[i] for i in range(nrow)]
    for x_idx in range(nrow):
        #axarr[x_idx, 0].set_ylabel(y_axis_labels[x_idx])  # Option 1
        axarr[x_idx, 0].set_ylabel(r'$x_{%d}$' % x_idx)  # Option 2
    axarr[-1, 0].set_xlabel(r'$t$')
    plt.suptitle('Solver comparison for single cell trajectory')

    # Plot kwargs
    alpha = 0.8
    ms = 4
    solver_to_kwargs = {
        0: dict(alpha=alpha, marker='s', linestyle='--', markersize=ms),
        1: dict(alpha=alpha, marker='o', linestyle='--', markersize=ms),
        2: dict(alpha=alpha, marker='^', linestyle='--', markersize=ms),
        3: dict(alpha=alpha, marker='*', linestyle='--', markersize=ms),
    }
    assert nsolvers <= len(solver_to_kwargs.keys())

    # Perform plotting
    for solver_idx in range(nsolvers):
        t = solver_times[solver_idx]
        traj = solver_traj[solver_idx]
        solver_label = solver_presets[solver_idx]['label']
        for x_idx in range(nrow):
            print(solver_idx, x_idx)
            #if x_idx > 0:
            #    solver_label = None
            axarr[x_idx, 0].plot(t, traj[x_idx, :], label=solver_label, **solver_to_kwargs[solver_idx])
        """
        if decorate:
            # plot points in phase space where division event occurred
            events_idx = self.time_indices_where_acted_as_mother(idx)
            for event_idx in events_idx:
                axarr[i, j].axvline(times[event_idx], linestyle='--', c='gray')
        """

    plt.legend()
    fpath = "output" + os.sep + "solver_comparison_singlecell"
    plt.savefig(fpath + '.pdf')
    plt.close()
    return traj_obj


if __name__ == '__main__':

    style_ode = 'PWL3_swap'  # PWL4_auto_linear, PWL3_swap, toy_clock
    sc = SingleCell(label='c1', style_ode=style_ode)
    if style_ode in ['PWL2', 'PWL3', 'PWL3_swap']:
        sc.params_ode['C'] = 1e-2
        sc.params_ode['epsilon'] = 0.37
        sc.params_ode['t_pulse_switch'] = 25

    solver_presets = [
        PRESET_SOLVER['solve_ivp_radau_default'],
        PRESET_SOLVER['solve_ivp_radau_minstep']
    ]

    compare_solvers_singlecell(sc, solver_presets)
