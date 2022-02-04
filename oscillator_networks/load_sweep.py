import os

from class_sweep_cellgraph import SweepCellGraph
from file_io import pickle_load


def visualize_sweep(sweep):

    print("Visualizing data for sweep label:", sweep.sweep_label)
    sweep.printer()

    for subdir in dir_sweep:
        pass
        # A) load classdump

        # B) extract useful info and add to collection

    # Visualize info

    return


if __name__ == '__main__':
    dir_sweep = 'runs' + os.sep + 'sweep_A'
    fpath_pickle = dir_sweep + os.sep + 'sweep.pkl'
    sweep_cellgraph = pickle_load(fpath_pickle)

    visualize_sweep(sweep_cellgraph)
