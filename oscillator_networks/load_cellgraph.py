import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from class_cellgraph import CellGraph
from file_io import pickle_load


if __name__ == '__main__':

    flag_print_state = True
    flag_replot = True
    flag_inspect = False
    flag_plotly = False
    flag_redetect = False

    runs_dir = 'runs' + os.sep + 'cellgraph'
    specific_dir = runs_dir + os.sep + '2022-02-03_05.32.09PM' #'2022-02-02_02.47.01PM'
    #specific_dir = 'input'

    # load classdump pickle file from "specific dir"
    fpath = specific_dir + os.sep + 'classdump.pkl'
    cellgraph = pickle_load(fpath)

    # Shorthands
    pp = cellgraph.sc_template.params_ode
    times = cellgraph.times_history
    state = cellgraph.state_history
    state_tensor = cellgraph.state_to_rectangle(state)
    div_events = cellgraph.division_events
    birthdays = cellgraph.cell_stats[:, 2]

    # Print method of CellGraph
    if flag_print_state:
        cellgraph.print_state()
        print(times[0:5])

    # "replot" standard cellgraph trajectory outputs with minor adjustments
    if flag_replot:
        cellgraph.plot_graph(fmod='replot', seed=0)
        cellgraph.plot_state_unified(fmod='replot', arrange_vertical=True)
        if cellgraph.sc_dim_ode > 1:
            cellgraph.plot_xy_separate(fmod='replot')

    # manual plot to inspect trajectory
    if flag_inspect:
        # plot time slice for one cell
        t0_idx = 13
        t1_idx = 136
        cell_choice = 0
        times_slice = times[t0_idx:t1_idx]
        state_slice = state_tensor[:, 0, t0_idx:t1_idx]
        for i, t in enumerate(times_slice):
            print(i, t, state_slice[:, i])

        plt.plot(times_slice, state_slice.T, 'o', label=['x%d' % i for i in range(cellgraph.sc_dim_ode)])
        # add any axhline decorators
        clow = 0.5*(pp['a'])  # None
        chigh = 0.5*(pp['a'] - pp['d'])  # None
        if clow is not None:
            plt.axhline(clow, linestyle='--', c='gray')
        if chigh is not None:
            plt.axhline(chigh, linestyle='--', c='gray')
        # decorate any division events in window
        events_idx = cellgraph.time_indices_where_acted_as_mother(cell_choice)
        for event_idx in events_idx:
            plt.axvline(times[event_idx], linestyle=':', c='r')
        plt.legend()
        plt.show()

    if flag_plotly:
        # Example dataframe
        #df = px.data.gapminder().query("country in ['Canada', 'Botswana']")
        #print(df)
        cellgraph.plotly_traj(fmod='replot', show=True, write=True)

    if flag_redetect:
        # reset division trackers (so that early ones can be identified)
        cellgraph.division_events = np.zeros((0, div_events.shape[1]))
        cellgraph.cell_stats[:, 1] = 0
        # re-detect division events
        event_detected, mother_cell, event_time_idx, times_history_truncated, state_history_truncated = \
            cellgraph.detect_oscillations_graph_trajectory(times, state)
        print("\nRE-DETECTION RESULTS: \nevent_detected, mother_cell, event_time_idx, abs time")
        print(event_detected, mother_cell, event_time_idx, times[event_time_idx])