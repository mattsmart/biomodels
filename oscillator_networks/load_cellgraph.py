import matplotlib.pyplot as plt
import numpy as np
import os

from class_cellgraph import CellGraph
from file_io import pickle_load


if __name__ == '__main__':

    flag_replot = False
    flag_inspect = True

    runs_dir = 'runs' + os.sep + 'cellgraph'
    specific_dir = runs_dir + os.sep + '2022-02-02_01.00.24PM' #'2022-02-01_03.20.31PM'
    fpath = specific_dir + os.sep + 'classdump.pkl'
    cellgraph = pickle_load(fpath)

    # "replot" standard cellgraph trajectory outputs with minor adjustments
    if flag_replot:
        cellgraph.plot_graph(fmod='replot', seed=17)
        cellgraph.plot_state_unified(fmod='replot', arrange_vertical=True)
        if cellgraph.sc_dim_ode > 1:
            cellgraph.plot_xy_separate(fmod='replot')

    # manual plot to inspect trajectory
    if flag_inspect:
        times = cellgraph.times_history
        state = cellgraph.state_history
        state_tensor = cellgraph.state_to_rectangle(state)
        div_events = cellgraph.division_events

        t0_idx = 25
        t1_idx = 95
        cell_choice = 0
        times_slice = times[t0_idx:t1_idx]
        state_slice = state_tensor[:, 0, t0_idx:t1_idx]
        plt.plot(times_slice, state_slice.T, 'o')
        # decorate any division events in window
        events_idx = cellgraph.time_indices_where_acted_as_mother(cell_choice)
        for event_idx in events_idx:
            plt.axvline(times[event_idx], linestyle='--', c='gray')
        plt.show()
