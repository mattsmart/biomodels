import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time

from preset_solver import PRESET_SOLVER
from class_singlecell import SingleCell


def compare_solvers_singlecell(single_cell, solver_presets, timer_mode=True, nrepeats=10):
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
        if timer_mode:
            timings = [0] * nrepeats
            for idx in range(nrepeats):
                t1 = time.time()
                r, times = sc.trajectory(flag_info=True, dynamics_method=solver_dynamics_method, **solver_kwargs)
                t2 = time.time()
                timings[idx] = t2-t1
            solver_presets[solver_idx]['timings'] = timings
        else:
            r, times = sc.trajectory(flag_info=True, dynamics_method=solver_dynamics_method, **solver_kwargs)
        solver_times[solver_idx] = times
        solver_traj[solver_idx] = r.T

    # Plot timing info if timer mode
    if timer_mode:
        column_names = ['solver', 'timing_index', 'timing']
        df = pd.DataFrame(columns=column_names)
        i = 0
        for idx in range(nrepeats):
            for solver_idx in range(nsolvers):
                row = [solver_presets[solver_idx]['label'],
                       idx,
                       solver_presets[solver_idx]['timings'][idx]]
                df.loc[i] = row
                i += 1
        plt.figure(figsize=(6, 6))
        print(df)
        colors = ['#78C850', '#F08030', '#6890F0', '#F8D030', '#F85888', '#705898', '#98D8D8']
        #boxplot = sns.boxplot(x=df["timing"], y=df["solver"], palette=colors)
        boxplot = sns.boxplot(y=df["timing"], x=df["solver"], palette=colors)
        boxplot.set_ylabel("Runtime (s)", fontsize=12)
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tight_layout()
        plt.gca().set_ylim([0, 0.3])
        plt.savefig("output" + os.sep + "solver_comparison_singlecell_timings.pdf", bbox_inches='tight')
        plt.show(bbox_inches='tight')
        plt.close()

    # Plotting trajectory from each solver
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
        1: dict(alpha=alpha, marker='o', linestyle='-.', markersize=ms),
        2: dict(alpha=alpha, marker='^', linestyle=':', markersize=ms),
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
    #plt.show()
    plt.close()
    return traj_obj


if __name__ == '__main__':

    style_ode = 'PWL3_swap'  # PWL4_auto_linear, PWL3_swap, toy_clock
    sc = SingleCell(label='c1', style_ode=style_ode)
    if style_ode in ['PWL2', 'PWL3', 'PWL3_swap']:
        sc.params_ode['epsilon'] = 1e-2
        sc.params_ode['pulse_vel'] = 0.37
        sc.params_ode['t_pulse_switch'] = 25

    solver_presets = [
        PRESET_SOLVER['solve_ivp_BDF_default'],
        PRESET_SOLVER['solve_ivp_radau_default'],
        PRESET_SOLVER['solve_ivp_radau_minstep'],
        PRESET_SOLVER['solve_ivp_radau_relaxed'],
    ]

    compare_solvers_singlecell(sc, solver_presets)
