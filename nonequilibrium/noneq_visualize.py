import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from noneq_functions import state_to_label, label_to_state, hamiltonian


# DICTIONARIES
N = 3
labels_to_states01 = {idx: label_to_state(idx, N, use_neg=False) for idx in xrange(2 ** N)}
states01_to_labels = {tuple(v): k for k, v in labels_to_states01.iteritems()}
states01_to_states = {state:tuple([2*v - 1 for v in state]) for state in states01_to_labels.keys()}

# DRAWING HYPERCUBES
G = nx.hypercube_graph(N)
pos = nx.spring_layout(G)

print "drawing original G"
nx.draw_networkx_nodes(G, pos=pos, nodelist = G.nodes())
nx.draw_networkx_edges(G, pos=pos, edgelist = G.edges())
nx.draw_networkx_labels(G, pos=pos, font_size=12)
plt.gca().axis('off')
plt.show()

print "nodes of G by default"
print G.nodes()

print "relabelling nodes to standard up/down +1/-1"
#nx.relabel_nodes(G, states01_to_states, copy=False)
#print G.nodes()

print "redrawing G with new nodes"
nx.draw_networkx_nodes(G, pos=pos, nodelist = G.nodes())
nx.draw_networkx_edges(G, pos=pos, edgelist = G.edges())
nx.draw_networkx_labels(G, pos=pos, font_size=12)
plt.gca().axis('off')
plt.show()


ensemble = [(1,1,0), (1,0,0), (1,1,0), (1,0,0), (1,1,0), (1,0,0), (1,1,0), (1,0,0)]
ensemble_pos = np.zeros((len(ensemble), 2))
mu = 1.0
sigma = 0.02
x_perturb = np.random.normal(mu, sigma, len(a))
y_perturb = np.random.normal(mu, sigma, len(a))
for idx, elem in ensemble:
    val_xy = ensemble_pos[elem]
    val_xy_perturbed = val_xy * np.array(x_perturb, y_perturb)
    ensemble_pos[idx,:] = val_xy_perturbed
print ensemble_pos
print
