from abstract_graph import PyAbstractGraph

import numpy as _np
import networkx as _nx

class PyCustomGraph(PyAbstractGraph):
	r"""A custom graph, specified by a list of edges and optionally colors."""

	def __init__(self, edges=[]):
		"""
		Constructs a new graph given a list of edges.

		Args:
		    edges: If `edges` has elements of type `Tuple[int, int]` it is treated
		        as a list of edges. Then each element `(i, j)` means a connection
		        between sites `i` and `j`. It is assumed that `0 <= i <= j`. Also,
		        `edges` should contain no duplicates. If `edges` has elements of
		        type `Tuple[int, int, int]` each element `(i, j, c)` represents an
		        edge between sites `i` and `j` colored into `c`. It is again assumed
		        that `0 <= i <= j` and that there are no duplicate elements in `edges`.


		Examples:
		    A 10-site one-dimensional lattice with periodic boundary conditions can be
		    constructed specifying the edges as follows:

		    >>> import netket
		    >>> g=netket.graph.PyCustomGraph([[i, (i + 1) % 10] for i in range(10)])
		    >>> print(g.n_sites)
		    10
		"""

		assert isinstance(edges, list)
		type_condition = [isinstance(edge, list) or isinstance(edge, tuple) for edge in edges]
		if False in type_condition:
			raise ValueError("edges must be a list of lists or tuples")

		edges_array = _np.array(edges, dtype=_np.int32)
		if edges_array.ndim != 2:
			raise ValueError("edges must be a list of lists or tuples of the same length (2 or 3)")

		if not(edges_array.shape[1] == 2 or edges_array.shape[1] == 3):
			raise ValueError("edges must be a list of lists or tuples of the same length (2 or 3), where the third column indicates the color")

		self.graph = _nx.MultiGraph()
		self.graph.add_edges_from(edges_array)

		super().__init__()
		
	def get_edges_from_color(self, color):
		# Could be handy to get all edges from a color
		return NotImplementedError

	@property
	def adjacency_list(self):
		return [list(self.graph.neighbors(node)) for node in self.graph.nodes]

	@property
	def is_connected(self):
		# TODO: how to check if a multigraph is connected?
		return _nx.is_connected(self.graph)

	@property
	def edges(self):
		return [e for e in self.graph.edges()]

	@property
	def n_sites(self):
		return self.graph.number_of_nodes()

	@property
	def distances(self):
		return _nx.floyd_warshall_numpy(self.graph)

	@property
	def is_bipartite(self):
		return _nx.is_bipartite(self.graph)

	@property
	def size(self):
		return self.graph.number_of_nodes()

	@property
	def edge_colors(self):
		return NotImplementedError

	@property
	def edges_and_colors(self):
		return NotImplementedError

	@property
	def automorphisms(self):
		# TODO: check if this is better than using the VF2 networkx algorithm
		# TODO: check how to compute these when we have a coloured graph where there could
		#       be a duplicated edge with two different colors.
		aux_graph = _nx.Graph(self.edges)
		ismags = _nx.isomorphism.ISMAGS(aux_graph, aux_graph)
		_authomorphisms = [[iso[i] for i in aux_graph.nodes()] for iso in ismags.isomorphisms_iter(symmetry=False)]
		return _authomorphisms