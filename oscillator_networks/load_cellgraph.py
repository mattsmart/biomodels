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

    flag_replot = False
    flag_inspect = False
    flag_plotly = True
    flag_redetect = False

    runs_dir = 'runs' + os.sep + 'cellgraph'
    specific_dir = runs_dir + os.sep + '2022-02-02_03.35.19PM' #'2022-02-02_02.47.01PM'
    fpath = specific_dir + os.sep + 'classdump.pkl'
    cellgraph = pickle_load(fpath)

    # shorthands
    pp = cellgraph.sc_template.params_ode
    times = cellgraph.times_history
    state = cellgraph.state_history
    state_tensor = cellgraph.state_to_rectangle(state)
    div_events = cellgraph.division_events
    birthdays = cellgraph.cell_stats[:, 2]

    # "replot" standard cellgraph trajectory outputs with minor adjustments
    if flag_replot:
        cellgraph.plot_graph(fmod='replot', seed=0)
        cellgraph.plot_state_unified(fmod='replot', arrange_vertical=True)
        if cellgraph.sc_dim_ode > 1:
            cellgraph.plot_xy_separate(fmod='replot')

    # manual plot to inspect trajectory
    if flag_inspect:
        # plot time slice for one cell
        t0_idx = 18
        t1_idx = 95
        cell_choice = 0
        times_slice = times[t0_idx:t1_idx]
        state_slice = state_tensor[:, 0, t0_idx:t1_idx]
        plt.plot(times_slice, state_slice.T, 'o')
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
            plt.axvline(times[event_idx], linestyle='--', c='gray')
        plt.show()

    if flag_plotly:
        # Example dataframe
        #df = px.data.gapminder().query("country in ['Canada', 'Botswana']")
        #print(df)

        column_names = ['cell', 'time_index', 'time'] + ['x%d' % i for i in range(cellgraph.sc_dim_ode)]
        df = pd.DataFrame(columns=column_names)
        i = 0
        for cell in range(cellgraph.num_cells):
            init_idx = birthdays[cell]
            looptot = len(times) - init_idx
            for idx, t in enumerate(range(init_idx, len(times))):
                row = ['cell%d' % cell,
                       t,
                       times[t]]
                row = row + [state_tensor[i, cell, t] for i in range(cellgraph.sc_dim_ode)]
                df.loc[idx + i] = row
            i += looptot

        cmap_list = px.colors.qualitative.Plotly
        fig = make_subplots(rows=cellgraph.sc_dim_ode,
                            cols=1,
                            x_title=r'$t$')

        for i in range(cellgraph.sc_dim_ode):
            if i == 0:
                showlegend = True
            else:
                showlegend = False
            for cell in range(cellgraph.num_cells):
                cell_color = cmap_list[cell % len(cmap_list)]
                fig.append_trace(
                    go.Scatter(
                        x=df[df['cell'] == 'cell%d' % cell]['time'],
                        y=df[df['cell'] == 'cell%d' % cell]['x%d' % i],
                        mode='lines+markers',
                        name='c%d' % (cell),
                        line=dict(color=cell_color),
                        marker_color=cell_color,
                        legendgroup=cell,
                        showlegend=showlegend,
                    ),
                    row=i + 1, col=1)
            fig.update_yaxes(title_text=cellgraph.sc_template.variables_short[i], row=i + 1, col=1)

        #fig.update_layout(height=600, width=600, title_text="Cell state trajectoriers for each variable")
        fig.update_layout(title_text="Cell state trajectoriers for each variable")
        fig.show()

    if flag_redetect:
        # reset division trackers (so that early ones can be identified)
        cellgraph.division_events = np.zeros((0, div_events.shape[1]))
        cellgraph.cell_stats[:, 1] = 0
        # re-detect division events
        event_detected, mother_cell, event_time_idx, times_history_truncated, state_history_truncated = \
            cellgraph.detect_oscillations_graph_trajectory(times, state)
        print("\nRE-DETECTION RESULTS: \nevent_detected, mother_cell, event_time_idx, abs time")
        print(event_detected, mother_cell, event_time_idx, times[event_time_idx])
