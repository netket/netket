from .abstract_graph import AbstractGraph
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
        if not (
            isinstance(graph, _nx.classes.graph.Graph)
            or isinstance(graph, _nx.classes.multigraph.MultiGraph)
        ):
            raise TypeError("graph must be a networx Graph or MultiGraph", type(graph))

        if isinstance(graph, _nx.classes.graph.Graph):
            self.graph = _nx.MultiGraph(graph)
        else:
            self.graph = graph

        self._automorphisms = None

        super().__init__()

    def adjacency_list(self):
        return [list(self.graph.neighbors(node)) for node in self.graph.nodes]

    def is_connected(self):
        return _nx.is_connected(self.graph)

    def nodes(self):
        return self.graph.nodes()

    def edges(self, color=False):
        if color is True:
            return self.graph.edges(data="color")
        elif color is not False:
            return ((u, v) for u, v, k in self.graph.edges(data="color") if k == color)
        else:  # color is False
            return self.graph.edges()

    def distances(self):
        return _nx.floyd_warshall_numpy(self.graph).tolist()

    def is_bipartite(self):
        return _nx.is_bipartite(self.graph)

    @property
    def n_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def n_edges(self):
        return self.graph.size()

    def automorphisms(self):
        # TODO: check how to compute these when we have a coloured graph where there could
        #       be a duplicated edge with two different colors.

        # For the moment, if there are colors, the method returns a NotImplementedError:
        colors = set(c for _, _, c in self.edges(color=True))
        if len(colors) >= 2:
            raise NotImplementedError(
                "automorphisms is not yet implemented for colored edges"
            )

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

    def __repr__(self):
        return "{}(n_nodes={})".format(
            str(type(self)).split(".")[-1][:-2], self.n_nodes
        )


def Graph(nodes=[], edges=[]):
    r"""
    Constructs a Graph given a list of nodes and edges.
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

    if edges:
        type_condition = [
            isinstance(edge, list) or isinstance(edge, tuple) for edge in edges
        ]
        if False in type_condition:
            raise ValueError("edges must be a list of lists or tuples")

        edges_array = _np.array(edges, dtype=_np.int32)
        if edges_array.ndim != 2:
            raise ValueError(
                "edges must be a list of lists or tuples of the same length (2 or 3)"
            )

        if not (edges_array.shape[1] == 2 or edges_array.shape[1] == 3):
            raise ValueError(
                "edges must be a list of lists or tuples of the same length (2 or 3), where the third column indicates the color"
            )

        # Sort node names for ordering reasons:
    if nodes:
        node_names = sorted(nodes)
    elif edges:
        node_names = sorted(set((node for edge in edges_array for node in edge)))

    graph = _nx.MultiGraph()
    graph.add_nodes_from(node_names)
    if edges:
        graph.add_edges_from(edges_array)
        if edges_array.shape[1] == 3:  # edges with color
            colors = {tuple(e): e[-1] for e in edges}
            _nx.set_edge_attributes(graph, colors, name="color")
        else:  # only one color
            _nx.set_edge_attributes(graph, 0, name="color")

    return NetworkX(graph)


def Edgeless(nodes):
    """
    Edgeless(nodes)

    Construct a set graph (collection of unconnected vertices).
    Args:
        nodes: An integer number of nodes or a list of ints that index nodes of a graph
    Example:
        A 10-site one-dimensional lattice with periodic boundary conditions can be
        constructed specifying the edges as follows:

        >>> import netket
        >>> g=netket.graph.Edgeless([0,1,2,3])
        >>> print(g.n_nodes)
        4
    """
    if not isinstance(nodes, list):
        if not isinstance(nodes, int):
            raise TypeError("nodes must be either an integer or a list")
        nodes = range(nodes)

    edgelessgraph = _nx.MultiGraph()
    edgelessgraph.add_nodes_from(nodes)

    return NetworkX(edgelessgraph)


def DoubledGraph(graph):
    """
    DoubledGraph(graph)

    Constructs a DoubledGraph representing the doubled hilbert space of a density operator.
    The resulting graph is composed of two disjoint sub-graphs identical to the input.
    """

    dedges = list(graph.edges())
    n_v = graph.n_nodes

    dedges += [(edge[0] + n_v, edge[1] + n_v) for edge in graph.edges()]

    return Graph(edges=dedges)


def disjoint_union(graph_1, graph_2):
    """
    disjoint_union(graph_1, graph_2)

    Args:
        graph_1: a NetworkX graph
        graph_2: a NetworkX graph

    Returns:
        The Disjoint union of the two graphs. See NetworkX documentation for more informations.
    """
    union_graph = _nx.disjoint_union(graph_1.graph, graph_2.graph)
    return NetworkX(union_graph)
