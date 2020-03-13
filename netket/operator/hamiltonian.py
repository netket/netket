from .._C_netket.operator import GraphOperator as _GraphOperator

import numpy as _np

def Ising(hilbert, h, J=1.0):
    """
    Constructs a new ``Ising`` given a hilbert space, a transverse field,
    and (if specified) a coupling constant.

    Args:
        hilbert: Hilbert space the operator acts on.
        h: The strength of the transverse field.
        J: The strength of the coupling. Default is 1.0.

    Examples:
        Constructs an ``Ising`` operator for a 1D system.

        >>> import netket as nk
        >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
        >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
        >>> op = nk.operator.Ising(h=1.321, hilbert=hi, J=0.5)
        >>> print(op.hilbert.size)
        20
    """
    sigma_x = _np.array([[0, 1], [1, 0]])
    sz_sz = _np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return _GraphOperator(hilbert, siteops=[-h * sigma_x], bondops=[J * sz_sz])


def Heisenberg(hilbert, J=1, sign_rule=None):
    """
    Constructs a new ``Heisenberg`` given a hilbert space.

    Args:
        hilbert: Hilbert space the operator acts on.
        J: The strength of the coupling. Default is 1.
        sign_rule: If enabled, Marshal's sign rule will be used. On a bipartite
                   lattice, this corresponds to a basis change flipping the Sz direction
                   at every odd site of the lattice. For non-bipartite lattices, the
                   sign rule cannot be applied. Defaults to True if the lattice is
                   bipartite, False otherwise.

    Examples:
     Constructs a ``Heisenberg`` operator for a 1D system.

        >>> import netket as nk
        >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
        >>> hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
        >>> op = nk.operator.Heisenberg(hilbert=hi)
        >>> print(op.hilbert.size)
        20
    """
    if sign_rule is None:
        sign_rule = hilbert.graph.is_bipartite

    sz_sz = _np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    exchange = _np.array([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
    if sign_rule:
        if not hilbert.graph.is_bipartite:
            raise ValueError("sign_rule=True specified for a non-bipartite lattice")
        heis_term = sz_sz - exchange
    else:
        heis_term = sz_sz + exchange
    return _GraphOperator(hilbert, bondops=[J * heis_term])
