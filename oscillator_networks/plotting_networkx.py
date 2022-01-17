import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_from_adjacency(A, draw_edge_labels=False):
    """
    create_using=nx.DiGraph -- store as directed graph with possible self-loops
    create_using=nx.DiGraph -- store as undirected graph with possible self-loops
    """
    G = nx.from_numpy_matrix(np.matrix(A), create_using=nx.Graph)
    layout = nx.spring_layout(G)
    nx.draw(G, layout)
    if draw_edge_labels:
        nx.draw_networkx_edge_labels(G, pos=layout)
    plt.show()


if __name__ == '__main__':
    A = [
        [0, 1, 0, .8, 0],
        [0, 0, .4, 0, .3],
        [0, 0, 0, 0, 0],
        [0, 0, .6, 0, .7],
        [0, 0, 0, .2, 0]
    ]

    A2 = [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ]

    draw_from_adjacency(A2)
