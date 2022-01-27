import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_from_adjacency(A, node_color=None, labels=True, draw_edge_labels=False):
    """
    create_using=nx.DiGraph -- store as directed graph with possible self-loops
    create_using=nx.DiGraph -- store as undirected graph with possible self-loops
    """
    cmap = 'Pastel1'  # options: 'plt.cm.Blues', 'Pastel1', 'Spectral_r'
    ns = 900
    alpha = 1.0
    font_color = 'k'  # options: 'whitesmoke', 'k'

    G = nx.from_numpy_matrix(np.matrix(A), create_using=nx.Graph)
    layout = nx.spring_layout(G)
    nx.draw(G, layout, node_color=node_color, cmap=cmap, node_size=ns, alpha=alpha)
    if labels is not None:
        #nx.draw_networkx_labels(G, layout, labels, font_size=8.5, font_color='k')
        nx.draw_networkx_labels(G, layout, labels, font_size=8, font_color=font_color)
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
