from graph import Graph

import numpy as _np
import networkx as _nx


class Edgeless(Graph):
    r"""A set graph (collection of unconnected vertices)."""

    def __init__(self, n_nodes):
        """
    	Constructs a new set of given number of vertices.

		Args:
		    n_nodes: The number of nodes.

		Examples:
		    A 10-site set:

		    >>> import netket
		    >>> g=netket.graph.Edgeless(10)
		    >>> print(g.n_nodes)
		    10
    	"""
        super().__init__([(i, i) for i in range(n_nodes)])
        edgelessgraph = _nx.MultiGraph()
        edgelessgraph.add_nodes_from(self.graph)
        self.graph = edgelessgraph
