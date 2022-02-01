from class_cellgraph import CellGraph
from file_io import pickle_load


if __name__ == '__main__':
    cellgraph = pickle_load('foo.pkl')
    cellgraph.plot_graph(fmod='replot', seed=2)
