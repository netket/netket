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

import abc
from typing import List, Iterator, Sequence, Tuple, Union, Optional


Edge = Tuple[int, int]
ColoredEdge = Tuple[int, int, int]
EdgeSequence = Union[Sequence[Edge], Sequence[ColoredEdge]]


class AbstractGraph(abc.ABC):
    """Abstract class for NetKet graph objects."""

    @abc.abstractmethod
    def is_connected(self) -> bool:
        r"""True if the graph is connected"""
        raise NotImplementedError

    @abc.abstractmethod
    def is_bipartite(self) -> bool:
        r"""True if the graph is bipartite"""
        raise NotImplementedError

    @abc.abstractmethod
    def edges(
        self, return_color: bool = False, filter_color: Optional[int] = None
    ) -> EdgeSequence:
        r"""Returns the sequence of edges of the graph.

        Arguments:
            return_color: If :code:`True`, return edges with added color information.
            filter_color: If set, return only edges of the given color.

        Returns:
            A sequence of edges as tuples :code:`(i, j)` or, if `return_color` is
            passed, a sequence of tuples :code:`(i, j, c)` with :code:`c` indicating
            the color of the respective edge.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def edge_colors(self) -> Sequence[int]:
        r"""Sequence of edge colors, in the order of the edges returned by
        :code:`self.edges`."""
        raise NotImplementedError

    @abc.abstractmethod
    def nodes(self) -> Iterator[int]:
        r"""Iterator over the nodes of the graph"""
        raise NotImplementedError

    @abc.abstractmethod
    def distances(self) -> List[List]:
        r"""List containing the distances between the nodes.
        The distance between unconnected nodes (no path exists between them) is set to -1
        """
        raise NotImplementedError

    @abc.abstractmethod
    def automorphisms(self):
        r"""Symmetry group containing the automorphisms of the graph"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_nodes(self) -> int:
        r"""The number of nodes (or vertices) in the graph"""
        raise NotImplementedError

    @property
    def n_edges(self) -> int:
        r"""The number of edges in the graph."""
        return len(self.edges())

    @abc.abstractmethod
    def adjacency_list(self) -> List[List]:
        r"""List containing the adjacency list of the graph where each node
        is represented by an integer in [0, n_nodes)"""
        raise NotImplementedError
