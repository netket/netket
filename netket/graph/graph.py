# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Generator, Union

import numpy as np
import networkx as _nx

from netket.utils.semigroup import Permutation

from .abstract_graph import AbstractGraph
from .symmetry import SymmGroup


class NetworkX(AbstractGraph):
    """ Wrapper for a networkx graph"""

    def __init__(self, graph: _nx.Graph):
        """
        Constructs a netket graph from a networkx graph.

        Args:
            graph: A networkx graph (might be a :class:`networkx.Graph` or a :class:`networkx.MultiGraph`)

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

    def adjacency_list(self) -> List[List]:
        return [list(self.graph.neighbors(node)) for node in self.graph.nodes]

    def is_connected(self) -> bool:
        return _nx.is_connected(self.graph)

    def nodes(self) -> Generator:
        return self.graph.nodes()

    def edges(self, color: Union[bool, int] = False) -> Generator:
        if color is True:
            return self.graph.edges(data="color")
        elif color is not False:
            return ((u, v) for u, v, k in self.graph.edges(data="color") if k == color)
        else:  # color is False
            return self.graph.edges()

    def distances(self) -> List[List]:
        return _nx.floyd_warshall_numpy(self.graph).tolist()

    def is_bipartite(self) -> bool:
        return _nx.is_bipartite(self.graph)

    @property
    def n_nodes(self) -> int:
        r"""The number of nodes (or vertices) in the graph"""
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        r"""The number of edges in the graph."""
        return self.graph.size()

    def automorphisms(self) -> SymmGroup:
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
                Permutation([iso[i] for i in aux_graph.nodes()])
                for iso in ismags.isomorphisms_iter()
            ]
            self._automorphisms = SymmGroup(_automorphisms, self)
            return self._automorphisms

    def __repr__(self):
        return "{}(n_nodes={})".format(
            str(type(self)).split(".")[-1][:-2], self.n_nodes
        )


def Graph(nodes: List = [], edges: List = []) -> NetworkX:
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

        edges_array = np.array(edges, dtype=np.int32)
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


def Edgeless(nodes: Union[list, int]) -> NetworkX:
    """
    Construct a set graph (collection of unconnected vertices).

    Args:
        nodes: An integer number of nodes or a list of ints that index nodes of a graph.

    Example:
        >>> import netket
        >>> g=netket.graph.Edgeless([0,1,2,3])
        >>> print(g.n_nodes)
        4
        >>> print(g.n_edges)
        0
    """
    if not isinstance(nodes, list):
        if not isinstance(nodes, int):
            raise TypeError("nodes must be either an integer or a list")
        nodes = range(nodes)

    edgelessgraph = _nx.MultiGraph()
    edgelessgraph.add_nodes_from(nodes)

    return NetworkX(edgelessgraph)


def DoubledGraph(graph: AbstractGraph) -> NetworkX:
    """
    DoubledGraph(graph)

    Constructs a DoubledGraph representing the doubled hilbert space of a density operator.
    The resulting graph is composed of two disjoint sub-graphs identical to the input.

    Args:
        graph: The graph to double
    """

    dedges = list(graph.edges())
    n_v = graph.n_nodes

    dnodes = [i for i in range(n_v)] + [i + n_v for i in range(n_v)]

    dedges += [(edge[0] + n_v, edge[1] + n_v) for edge in graph.edges()]

    return Graph(nodes=dnodes, edges=dedges)


def disjoint_union(graph_1: NetworkX, graph_2: NetworkX) -> NetworkX:
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
