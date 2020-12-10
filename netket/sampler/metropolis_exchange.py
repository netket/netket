from .metropolis_hastings import *
from ._kernels import _ExchangeKernel
import numpy as _np


def _compute_clusters(graph, d_max):
    clusters = []
    distances = _np.asarray(graph.distances())
    size = distances.shape[0]
    for i in range(size):
        for j in range(i + 1, size):
            if distances[i][j] <= d_max:
                clusters.append((i, j))

    res_clusters = _np.empty((len(clusters), 2), dtype=_np.int64)

    for i, cluster in enumerate(clusters):
        res_clusters[i] = _np.asarray(cluster)

    return res_clusters


def MetropolisExchange(
    machine, clusters=None, graph=None, d_max=1, n_chains=16, sweep_size=None, **kwargs
):
    r"""
    This sampler acts locally only on two local degree of freedom :math:`s_i` and :math:`s_j`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s^\prime_j \dots s_N`,
    where in general :math:`s^\prime_i \neq s_i` and :math:`s^\prime_j \neq s_j`.
    The sites :math:`i` and :math:`j` are also chosen to be within a maximum graph
    distance of :math:`d_{\mathrm{max}}`.

    The transition probability associated to this sampler can
    be decomposed into two steps:

    1. A pair of indices :math:`i,j = 1\dots N`, and such
       that :math:`\mathrm{dist}(i,j) \leq d_{\mathrm{max}}`,
       is chosen with uniform probability.
    2. The sites are exchanged, i.e. :math:`s^\prime_i = s_j` and :math:`s^\prime_j = s_i`.

    Notice that this sampling method generates random permutations of the quantum
    numbers, thus global quantities such as the sum of the local quantum numbers
    are conserved during the sampling.
    This scheme should be used then only when sampling in a
    region where :math:`\sum_i s_i = \mathrm{constant}` is needed,
    otherwise the sampling would be strongly not ergodic.

    Args:
          machine: A machine :math:`\Psi(s)` used for the sampling.
                   The probability distribution being sampled
                   from is :math:`F(\Psi(s))`, where the function
                   :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.

          d_max: The maximum graph distance allowed for exchanges.
          n_chains: The number of Markov Chain to be run in parallel on a single process.
          sweep_size: The number of exchanges that compose a single sweep.
                      If None, sweep_size is equal to the number of degrees of freedom (n_visible).


    Examples:
          Sampling from a RBM machine in a 1D lattice of spin 1/2, using
          nearest-neighbours exchanges.

          >>> import netket as nk
          >>>
          >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
          >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
          >>>
          >>> # RBM Spin Machine
          >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
          >>>
          >>> # Construct a MetropolisExchange Sampler
          >>> sa = nk.sampler.MetropolisExchange(machine=ma)
          >>> print(sa.machine.hilbert.size)
          100
    """
    if clusters is None and graph is not None:
        assert (
            graph.n_nodes == machine.hilbert.size
        ), "The size of the graph must match the hilbert space of the machine."
        clusters = _compute_clusters(graph, d_max)
    elif not (clusters is not None and graph is None):
        raise ValueError(
            """You must either provide the list of exchange-clusters or a netket graph, from
                          which clusters will be computed using the maximum distance d_max. """
        )

    transition_kernel = _ExchangeKernel(machine, clusters)

    return MetropolisHastings(
        machine, transition_kernel, n_chains, sweep_size, **kwargs
    )


def MetropolisExchangePt(
    machine,
    clusters=None,
    graph=None,
    d_max=1,
    n_replicas=16,
    sweep_size=None,
    **kwargs,
):
    r"""
    This sampler performs parallel-tempering
    moves in addition to the local moves implemented in `MetropolisExchange`.
    The number of replicas can be chosen by the user.

    Args:
        machine: A machine :math:`\Psi(s)` used for the sampling.
                 The probability distribution being sampled
                 from is :math:`F(\Psi(s))`, where the function
                 :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
        d_max: The maximum graph distance allowed for exchanges.
        n_replicas: The number of replicas used for parallel tempering.
        sweep_size: The number of exchanges that compose a single sweep.
                    If None, sweep_size is equal to the number of degrees of freedom (n_visible).

    Examples:
        Sampling from a RBM machine in a 1D lattice of spin 1/2, using
        nearest-neighbours exchanges.

        >>> import netket as nk
        >>>
        >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
        >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
        >>>
        >>> # RBM Spin Machine
        >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
        >>>
        >>> # Construct a MetropolisExchange Sampler with parallel tempering
        >>> sa = nk.sampler.MetropolisExchangePt(machine=ma,n_replicas=24)
        >>> print(sa.machine.hilbert.size)
        100
    """
    if clusters is None and graph is not None:
        assert (
            graph.n_nodes == machine.hilbert.size
        ), "The size of the graph must match the hilbert space of the machine."

        clusters = _compute_clusters(graph, d_max)
    elif not (clusters is not None and graph is None):
        raise ValueError(
            """You must either provide the list of exchange-clusters or a netket graph, from
                          which clusters will be computed using the maximum distance d_max. """
        )

    transition_kernel = _ExchangeKernel(machine, clusters)

    return MetropolisHastingsPt(
        machine, transition_kernel, n_replicas, sweep_size, **kwargs
    )
