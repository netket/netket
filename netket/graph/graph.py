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

from typing import List, Optional, Sequence, Union

import numpy as np
import igraph

from netket.utils.deprecation import warn_deprecation
from netket.utils.group import Permutation, PermutationGroup
from .abstract_graph import AbstractGraph, Edge, ColoredEdge, EdgeSequence


class Graph(AbstractGraph):
    """
    A simple implementation of Graph based on an external graph library.

    The underlying implemnetation is based on igraph and supports conversion to
    networkx, but this is an implementation detail and could be changed in the future.
    """

    # Initialization
    # ------------------------------------------------------------------------
    def __init__(
        self,
        edges: Union[Sequence[Edge], Sequence[ColoredEdge]],
        n_nodes: Optional[int] = None,
    ):
        """
        Construct the a graph starting from a list of edges and optionally a given
        number of nodes.

        Args:
            edges: list of (undirected) edges
            n_nodes: number of nodes. Can be used to specify the number vertices in the
                graph if not all vertices appear in an edge.

        """
        edges, colors = self._clean_edges(edges)
        if n_nodes is None:
            if len(edges) > 0:
                n_nodes = max(max(e) for e in edges) + 1
            else:
                n_nodes = 0
        graph = igraph.Graph(directed=False)
        graph.add_vertices(n_nodes)
        graph.add_edges(edges, attributes={"color": colors})

        self._igraph = graph
        self._automorphisms = None

    @staticmethod
    def _clean_edges(edges):
        """Validate and normalize edges argument."""
        if not isinstance(edges, Sequence):
            raise TypeError("edges must be a sequence.")
        if len(edges) == 0:
            return edges, []

        e0 = edges[0]
        if not isinstance(e0, Sequence) or len(e0) not in (2, 3):
            raise ValueError(
                "Edges must be tuple of length 2 (or 3 for colored edges)."
            )
        if not all(len(e) == len(e0) for e in edges):
            raise ValueError("Either all or none of the edges need to specify a color.")

        if len(e0) == 2:
            return edges, [0] * len(edges)
        else:
            return [(v, w) for (v, w, _) in edges], [c for (*_, c) in edges]

    # Conversion
    # ------------------------------------------------------------------------
    @classmethod
    def from_igraph(cls, graph: igraph.Graph) -> "Graph":
        """
        Creates a new Graph instance from an igraph.Graph instance.
        """
        if graph.is_directed():
            raise ValueError("Directed graphs are not currently supported.")
        self = cls.__new__(cls)
        self._igraph = graph.copy()
        self._automorphisms = None

        if "color" not in self._igraph.edge_attributes():
            self._igraph.es.set_attribute_values("color", [0] * self._igraph.ecount())
        else:
            if not all(isinstance(c, int) for c in self.edge_colors):
                raise ValueError(
                    "graph has 'color' edge attributes, but not all colors are integers."
                )

        return self

    @classmethod
    def from_networkx(cls, graph) -> "Graph":
        """
        Creates a new Graph instance from a networkx graph.
        """
        ig = igraph.Graph.from_networkx(graph)
        return cls.from_igraph(ig)

    def to_igraph(self):
        """
        Returns a copy of this graph as an igraph.Graph instance.
        """
        return self._igraph.copy()

    def to_networkx(self):
        """
        Returns a copy of this graph as an igraph.Graph instance.
        This method requires networkx to be installed.
        """
        return self._igraph.to_networkx()

    # Graph properties
    # ------------------------------------------------------------------------
    def adjacency_list(self) -> List[List]:
        return self._igraph.get_adjlist()

    def is_connected(self) -> bool:
        return self._igraph.is_connected()

    def is_bipartite(self) -> bool:
        return self._igraph.is_bipartite()

    @property
    def n_nodes(self) -> int:
        r"""The number of nodes (or vertices) in the graph"""
        return self._igraph.vcount()

    @property
    def n_edges(self):
        r"""The number of edges in the graph."""
        return self._igraph.ecount()

    def nodes(self) -> Sequence[int]:
        return range(self._igraph.vcount())

    def edges(
        self,
        color=None,
        *,
        return_color: bool = False,
        filter_color: Optional[int] = None,
    ) -> EdgeSequence:
        if color is not None:
            warn_deprecation(
                "The color option has been split into return_color and filter_color."
            )
            # need to check for bool first, because bool is a subclass of int
            if isinstance(color, bool):
                return_color = color
            elif isinstance(color, int):
                filter_color = color
            else:
                raise TypeError("Incorrect type for 'color'")

        if not return_color and filter_color is None:
            return self._igraph.get_edgelist()

        edges_with_color = zip(self._igraph.get_edgelist(), self.edge_colors)
        if filter_color is not None:
            edges_with_color = filter(lambda x: x[1] == filter_color, edges_with_color)
        if return_color:
            return [(*e, c) for (e, c) in edges_with_color]
        else:
            return [e for (e, _) in edges_with_color]

    @property
    def edge_colors(self) -> Sequence[int]:
        r"""Sequence of edge colors, in the order of the edges returned by
        :code:`self.edges`."""
        if self.n_edges > 0:
            return self._igraph.es.get_attribute_values("color")
        else:
            return []

    # Graph algorithms
    # ------------------------------------------------------------------------
    def distances(self) -> List[List]:
        return np.array(self._igraph.shortest_paths())

    def _compute_automorphisms(self):
        """
        Compute the graph autmorphisms of this graph.
        """
        colors = self.edge_colors
        result = self._igraph.get_isomorphisms_vf2(
            edge_color1=colors, edge_color2=colors
        )

        # sort them s.t. the identity comes first
        result = np.unique(result, axis=0).tolist()
        result = PermutationGroup([Permutation(i) for i in result], self.n_nodes)
        return result

    # TODO turn into a struct.property_cached?
    def automorphisms(self) -> PermutationGroup:
        if self._automorphisms is None:
            self._automorphisms = self._compute_automorphisms()
        return self._automorphisms

    # Output and drawing
    # ------------------------------------------------------------------------
    def __repr__(self):
        return "{}(n_nodes={}, n_edges={})".format(
            str(type(self)).split(".")[-1][:-2], self.n_nodes, self.n_edges
        )


def Edgeless(n_nodes: int) -> Graph:
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
    return Graph([], n_nodes)


def DoubledGraph(graph: AbstractGraph) -> Graph:
    """
    DoubledGraph(graph)

    Constructs a DoubledGraph representing the doubled hilbert space of a density
    operator. The resulting graph is composed of two disjoint sub-graphs identical
    to the input.

    Args:
        graph: The graph to double
    """

    dedges = list(graph.edges())
    n_v = graph.n_nodes

    dnodes = 2 * n_v
    dedges += [(edge[0] + n_v, edge[1] + n_v) for edge in graph.edges()]

    return Graph(n_nodes=dnodes, edges=dedges)


def disjoint_union(graph_1: Graph, graph_2: Graph) -> Graph:
    """
    Returns the disjoint union of two graphs.
    """
    return Graph.from_igraph(igraph.disjoint_union([graph_1._igraph, graph_2._igraph]))
