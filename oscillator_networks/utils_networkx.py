import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism.tree_isomorphism import rooted_tree_isomorphism, tree_isomorphism


def draw_from_adjacency(A, node_color=None, labels=None, draw_edge_labels=False, cmap='Pastel1', title='Cell graph',
                        spring=False, seed=None, fpath=None, figsize=(4,4)):
    """
    create_using=nx.DiGraph -- store as directed graph with possible self-loops
    create_using=nx.DiGrapsh -- store as undirected graph with possible self-loops

    cmap options: 'Blues', 'Pastel1', 'Spectral_r'

    Note: observed issues on Windows when sweeps performed with draw_from_adjacency(spring=False) on each run
    ====== Process finished with exit code -1073740791 (0xC0000409) ======
        - think too many calls to layout = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
    """
    # TODO alternative visualization wth legend for discrete data (nDiv) and colorbar for continuous data (birth times)

    def pick_seed_using_num_cells():
        seed_predefined = {
            1: 0,
            2: 0,
            4: 0,
            8: 0,
            16: 0,
            32: 0,
        }
        seed = seed_predefined.get(A.shape[0], 0)  # M = A.shape[0] and seed_default = 0
        return seed

    # plot settings
    ns = 400  # 800
    alpha = 1.0
    fs = 10  # 6
    if labels is not None:
        fs = 6
    font_color = 'k'  # options: 'whitesmoke', 'k'
    if seed is None:
        seed = pick_seed_using_num_cells()

    # initialize the figure
    plt.figure(figsize=figsize)  # default 8,8; try 4,4 for quarter slide, or 6,6 for half a slide
    ax = plt.gca()
    ax.set_title(title)

    # initialize the graph
    G = nx.from_numpy_matrix(np.matrix(A), create_using=nx.Graph)
    # determine node positions
    if spring:
        layout = nx.spring_layout(G, seed=seed)
    else:
        # prog options: twopi, circo, dot
        layout = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")

    # draw the nodes
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    #nx.draw(G, layout, node_color=node_color, cmap=cmap, node_size=ns, alpha=alpha)
    nx.draw_networkx(G, layout, with_labels=False,
                     node_color=node_color, cmap=cmap, node_size=ns, alpha=alpha,
                     width=1.0, linewidths=2.0, edgecolors='black')
    # write node labels
    #cell_labels = {idx: r'$c_{%d}$' % (idx) for idx in range(A.shape[0])}
    #cell_labels = {idx: r'Cell $%d$' % (idx) for idx in range(A.shape[0])}
    cell_labels = {idx: r'$%d$' % (idx) for idx in range(A.shape[0])}
    if labels is not None:
        nx.draw_networkx_labels(G, layout, labels, font_size=fs, font_color=font_color, verticalalignment='bottom')
        cell_labels = {idx: r'Cell $%d$' % (idx) for idx in range(A.shape[0])}
        nx.draw_networkx_labels(G, layout, cell_labels, font_size=fs, font_color=font_color, verticalalignment='top')
    else:
        cell_labels = {idx: r'$%d$' % (idx) for idx in range(A.shape[0])}
        nx.draw_networkx_labels(G, layout, cell_labels, font_size=fs, font_color=font_color)
    # write edge labels
    if draw_edge_labels:
        nx.draw_networkx_edge_labels(G, pos=layout)

    ax.axis("off")
    if fpath is None:
        plt.show()
    else:
        plt.savefig(fpath)
    plt.close()  # TODO to avoid 20 figures open warning
    return ax


def check_tree_isomorphism(A1, A2, root1=None, root2=None, rooted=False):
    """
    See documentation here: (fast methods for checking tree type graph ismorphism
    https://networkx.org/documentation/stable/reference/algorithms/isomorphism.html?highlight=isomorphism#module-networkx.algorithms.isomorphism.tree_isomorphism
    """
    G1 = nx.from_numpy_matrix(np.matrix(A1), create_using=nx.Graph)
    G2 = nx.from_numpy_matrix(np.matrix(A2), create_using=nx.Graph)
    if rooted:
        print("WARNING - Use tree_isomorphism, not the rooted variant, as it seems to give incorrect results")
        # issue 2022_0207 - why is this giving diff results from tree_isomorphism?
        iso_list = rooted_tree_isomorphism(G1, root1, G2, root2)
    else:
        iso_list = tree_isomorphism(G1, G2)

    if not iso_list:
        is_isomorphic = False  # i.e. it is an empty list, so no isomorphism found
    else:
        is_isomorphic = True
    return is_isomorphic, iso_list

if __name__ == '__main__':

    flag_draw = False
    flag_isomorphism_check = True

    if flag_draw:
        A1 = np.array([
            [0, 1, 0, .8, 0],
            [0, 0, .4, 0, .3],
            [0, 0, 0, 0, 0],
            [0, 0, .6, 0, .7],
            [0, 0, 0, .2, 0]
        ])

        A2 = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])
        draw_from_adjacency(A2, fpath='foo.pdf')

    if flag_isomorphism_check:
        #from networkx.algorithms import isomorphism
        # isomorphism.tree_isomorphism.rooted_tree_isomorphism(A1, 0, A2, 0)

        A1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])

        A2_iso_to_A1 = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 0]
        ])

        A3_distinct = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ])



        print("Empty list means its found to be NOT isomorphic (no mapping)")
        print('Check A1, A1 - trivially isomorphic')
        print(check_tree_isomorphism(A1, A1, root1=0, root2=0, rooted=True))
        print(check_tree_isomorphism(A1, A1))
        print('Check A1, A2 - expect isomorphism by swap node 0 and node 1')
        print(check_tree_isomorphism(A1, A2_iso_to_A1, root1=0, root2=0, rooted=True))
        print(check_tree_isomorphism(A1, A2_iso_to_A1))
        print('Repeat for A1, A3 - expect distinct')
        print(check_tree_isomorphism(A1, A3_distinct, root1=0, root2=0, rooted=True))
        print(check_tree_isomorphism(A1, A3_distinct))
