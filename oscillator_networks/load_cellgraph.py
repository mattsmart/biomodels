import os

from class_cellgraph import CellGraph
from file_io import pickle_load


if __name__ == '__main__':
    runs_dir = 'runs' + os.sep + 'cellgraph'
    specific_dir = runs_dir + os.sep + '2022-02-01_05.08.39PM' #'2022-02-01_03.20.31PM'
    fpath = specific_dir + os.sep + 'classdump.pkl'

    cellgraph = pickle_load(fpath)
    cellgraph.plot_graph(fmod='replot', seed=2)
    cellgraph.plot_state_unified(fmod='replot', arrange_vertical=True)
    if cellgraph.sc_dim_ode > 1:
        cellgraph.plot_xy_separate(fmod='replot')
