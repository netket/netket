from ._C_netket.hilbert import *


def Qubit(graph):
    """
    Constructs a new ``Qubit`` given a graph.

    Args:
        graph: Graph representation of sites.

    Examples:
        Simple qubit hilbert space.

        >>> from netket.graph import Hypercube
        >>> from netket.hilbert import Qubit
        >>> g = Hypercube(length=10,n_dim=2,pbc=True)
        >>> hi = Qubit(graph=g)
        >>> print(hi.size)
        100
    """
    return CustomHilbert(graph, local_states=[0, 1])
