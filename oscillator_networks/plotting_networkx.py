import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_from_adjacency(A, node_color=None, labels=None, draw_edge_labels=False, cmap='Pastel1', title='Cell graph', fpath=None):
    """
    create_using=nx.DiGraph -- store as directed graph with possible self-loops
    create_using=nx.DiGrapsh -- store as undirected graph with possible self-loops

    cmap options: 'Blues', 'Pastel1', 'Spectral_r'
    """
    # plot settings
    ns = 800
    alpha = 1.0
    font_color = 'k'  # options: 'whitesmoke', 'k'

    # initialize the figure
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_title(title)

    # initialize the graph
    G = nx.from_numpy_matrix(np.matrix(A), create_using=nx.Graph)
    # determine node positions
    seed = 1
    layout = nx.spring_layout(G, seed=seed)
    # draw the nodes
    nx.draw(G, layout, node_color=node_color, cmap=cmap, node_size=ns, alpha=alpha)
    # write node labels
    if labels is not None:
        nx.draw_networkx_labels(G, layout, labels, font_size=8, font_color=font_color)
    # write edge labels
    if draw_edge_labels:
        nx.draw_networkx_edge_labels(G, pos=layout)

    if fpath is None:
        plt.show()
    else:
        plt.savefig(fpath)
    return ax


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

    draw_from_adjacency(A2, fpath='foo.pdf')
