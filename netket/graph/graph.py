from abstract_graph import AbstractGraph

import numpy as _np
import networkx as _nx

class NetworkX(AbstractGraph):
    """ Wrapper for a networkx graph"""
    def __init__(self, graph):
        """
        Constructs a netket graph from a networkx graph.

        Args: 
            graph: A networkx graph (might be a networkx.Graph or a networkx.MultiGraph)
        Examples:
            A graph of nodes [0,1,2] with edges [(0,1), (0,2), (1,2)]
            >>> import netket
            >>> import networkx
            >>> nx_g = networkx.Graph([(0,1), (0,2), (1,2)])
            >>> nk_g = netket.graph.NetworkX(nx_g)
            >>> print(nk_g.n_nodes)
            3
        """
        assert isinstance(graph, _nx.classes.graph.Graph) or isinstance(graph, _nx.classes.multigraph.MultiGraph)

        if isinstance(graph, _nx.classes.graph.Graph):
            self.graph = _nx.MultiGraph(graph)

        self._automorphisms = None

        super().__init__()

    @property
    def adjacency_list(self):
        return [list(self.graph.neighbors(node)) for node in self.graph.nodes]

    @property
    def is_connected(self):
        # TODO: how to check if a multigraph is connected?
        return _nx.is_connected(self.graph)

    def edges(self, color=False):
        if color is True:
            return list(self.graph.edges(keys=True))
        elif color is not False:
            return [(u, v) for u, v, k in self.graph.edges if k == color]
        else:
            return list(self.graph.edges(keys=False))

    def distances(self):
        return _nx.floyd_warshall_numpy(self.graph)

    @property
    def is_bipartite(self):
        return _nx.is_bipartite(self.graph)

    @property
    def size(self):
        print("DeprecationWarning: size is deprecated. Use n_nodes from now on.")
        return self.n_nodes

    @property
    def n_nodes(self):
        return self.graph.number_of_nodes()

    def automorphisms(self):
        # TODO: check how to compute these when we have a coloured graph where there could
        #       be a duplicated edge with two different colors.

        # For the moment, if there are colors, the method returns a NotImplementedError:
        if self.edges():
            colors = _np.unique(_np.array(self.edges(color=True))[:, 2])
        else:
            colors = _np.array([])
        if colors.size >= 2:
            return NotImplementedError

        if self._automorphisms is not None:
            return self._automorphisms
        else:
            aux_graph = _nx.Graph()
            aux_graph.add_nodes_from(self.graph.nodes())
            aux_graph.add_edges_from(self.edges())
            ismags = _nx.isomorphism.GraphMatcher(aux_graph, aux_graph)
            _automorphisms = [
                [iso[i] for i in aux_graph.nodes()]
                for iso in ismags.isomorphisms_iter()
            ]
            self._automorphisms = _automorphisms
            return _automorphisms

def Graph(nodes=[], edges=[]):
    """ A Custom Graph provided nodes or edges.
        Constructs a Custom Graph given a list of nodes and edges.
        Args:
            nodes: A list of ints that index nodes of a graph
            edges: A list of 2- or 3-tuples that denote an edge with an optional color

        The Graph can be constructed specifying only the edges and the nodes will be deduced from the edges.

        Examples:
            A 10-site one-dimensional lattice with periodic boundary conditions can be
            constructed specifying the edges as follows:

            >>> import netket
            >>> g=netket.graph.Graph(edges=[[i, (i + 1) % 10] for i in range(10)])
            >>> print(g.n_nodes)
            10

    """
    if not isinstance(nodes, list):
        raise TypeError("nodes must be a list")

    if not isinstance(edges, list):
        raise TypeError("edges must be a list")

    if not edges:
        return Edgeless(nodes)

    type_condition = [
        isinstance(edge, list) or isinstance(edge, tuple) for edge in edges
    ]
    if False in type_condition:
        raise TypeError("edges must be a list of lists or tuples")

    edges_array = _np.array(edges, dtype=_np.int32)
    if edges_array.ndim != 2:
        raise TypeError(
            "edges must be a list of lists or tuples of the same length (2 or 3)"
        )

    if not (edges_array.shape[1] == 2 or edges_array.shape[1] == 3):
        raise TypeError(
            "edges must be a list of lists or tuples of the same length (2 or 3), where the third column indicates the color"
        )

    # Sort node names for ordering reasons:
    if nodes:
        node_names = sorted(nodes)
    else:
        node_names = sorted(set((node for edge in edges_array for node in edge)))

    graph = _nx.MultiGraph()
    graph.add_nodes_from(node_names)
    graph.add_edges_from(edges_array)
    return NetworkX(graph)

def Edgeless(nodes):
    """A set graph (collection of unconnected vertices).
        Args:
            nodes: A list of ints that index nodes of a graph
        Example:
            A 10-site one-dimensional lattice with periodic boundary conditions can be
            constructed specifying the edges as follows:

            >>> import netket
            >>> g=netket.graph.Edgeless([0,1,2,3])
            >>> print(g.n_nodes)
            4
    """
    if not isinstance(nodes, list):
        raise TypeError("nodes must be a list")

    edgelessgraph = _nx.MultiGraph()
    edgelessgraph.add_nodes_from(nodes)
    return NetworkX(edgelessgraph)
