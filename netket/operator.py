from ._C_netket.operator import *
from ._C_netket.operator import _local_values_kernel

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
    return GraphOperator(hilbert, siteops=[-h * sigma_x], bondops=[J * sz_sz])


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
    return GraphOperator(hilbert, bondops=[J * heis_term])


def local_values(op, machine, samples, log_val_samples=None, out=None):
    """
    Computes local values of the operator `op` for all `samples`.

    The local value is defined as
    .. math:: O_{\mathrm{loc}}(x) = \langle x | O | \Psi \rangle / \langle x | \Psi \rangle


            Args:
                op: Hermitian operator.
                samples: A matrix X containing a batch of visible
                    configurations :math:`x_1,\dots x_M`.
                    Each row of the matrix corresponds to a visible configuration.
                machine: Wavefunction :math:`\Psi`.
                log_val_samples: An array containing the values :math:`\Psi(X)`.
                    If not given, it is computed from scratch.
                    Defaults to None.
                out: A numpy array of local values of the operator.
                    If not given, it is allocated from scratch and then returned.
                    Defaults to None.     

            Returns:
                A numpy array of local values of the operator.
    """
    if out is None:
        out = _np.zeros(shape=samples.shape[0:-1], dtype=_np.complex128)

    if log_val_samples is None:
        log_val_samples = machine.log_val(samples)

    vprimes, mels = op.get_conn(samples)
    log_val_primes = [machine.log_val(vprime) for vprime in vprimes]

    _local_values_kernel(log_val_samples, log_val_primes, mels, out)

    # for k, sample in enumerate(samples):
    #
    #     lvd = machine.log_val(vprimes[k])
    #
    #     # out[k] = (mels[k] * _np.exp(lvd - log_val_samples[k])).sum()

    return out
