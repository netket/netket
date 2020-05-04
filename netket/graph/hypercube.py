from custom_graph import PyCustomGraph

import numpy as _np
import networkx as _nx

class PyHypercube(PyCustomGraph):
	r"""A hypercube lattice of side L in d dimensions. 
	Periodic boundary conditions can also be imposed"""

	def __init__(self, length, n_dim=1, pbc=False):
		"""
		Constructs a new ``Hypercube`` given its side length and dimension.

		Args:
			length: Side length of the hypercube.
				It must always be >= 1,
				but if ``pbc==True`` then the minimal valid length is 3.
				It can also be a list[int] where each entry is the size of a dimension.
			n_dim: Dimension of the hypercube. It must be at least 1.
			pbc: If ``True```then the constructed hypercube will have
				periodic boundary conditions, otherwise open boundary conditions
				are imposed.
		Examples:
		    A 10x10 square lattice with periodic boundary conditions can be
		    constructed as follows:

		    >>> import netket
		    >>> g=netket.graph.PyHypercube(length=10,n_dim=2,pbc=True)
		    >>> print(g.n_sites)
		    100

		    Also, a 2x2x3 lattice with open boundary conditions can be constructed as follows:
		    >>> g=netket.graph.PyHypercube(length=[2,2,3], pbc=False)
		    >>> print(g.n_sites)
		    12
		"""

		assert isinstance(length, int) or isinstance(length, list)
		assert isinstance(n_dim, int) and n_dim >= 1

		if isinstance(length, int):
			length = [length] * n_dim
		else:
			try:
				condition = [isinstance(x, int) and x >= 1 for x in length]
				if sum(condition) != len(length):
					raise TypeError
			except TypeError:
				raise TypeError("Components of length must be integers greater than 1")

		graph = _nx.generators.lattice.grid_graph(length, periodic=pbc)
		newnames = {old: new for new, old in enumerate(graph.nodes)}
		graph = _nx.relabel_nodes(graph, newnames)
		super().__init__(list(graph.edges))