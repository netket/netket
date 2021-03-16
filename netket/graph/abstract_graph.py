import abc

from typing import List, Generator, Iterator, Tuple


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
    def edges(self) -> Iterator[Tuple[int, int]]:
        r"""Iterator over the edges of the graph"""
        raise NotImplementedError

    @abc.abstractmethod
    def nodes(self) -> Iterator[int]:
        r"""Iterator over the nodes of the graph"""
        raise NotImplementedError

    @abc.abstractmethod
    def distances(self) -> List[List]:
        r"""List containing the distances between the nodes.
        The fact that some node may not be reachable from another is represented by -1"""
        raise NotImplementedError

    @abc.abstractmethod
    def automorphisms(self) -> List[List]:
        r"""List containing the automorphisms of the graph"""
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
