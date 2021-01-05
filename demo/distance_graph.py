import networkx as nx
import numpy as np
import string

import matplotlib.pyplot as plt

A = np.array([(0, 0.3, 0.4, 0.7),
              (0.3, 0, 0.9, 0.2),
              (0.4, 0.9, 0, 0.1),
              (0.7, 0.2, 0.1, 0)])

# A= np.load('../engine/qf.npy')
G = nx.from_numpy_matrix(A)
G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))
nx.draw(G)
plt.show()
# H = nx.path_graph(10)
# G.add_nodes_from(H)
# nx.draw(G, with_labels=True)
# plt.show()

import matplotlib.pyplot as plt
import networkx as nx

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")

G = nx.balanced_tree(3, 5)
pos = graphviz_layout(G, prog='twopi', args='')
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
plt.axis('equal')
plt.show()