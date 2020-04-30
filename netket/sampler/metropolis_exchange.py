import numpy as _np
from netket import random as _random

from .abstract_sampler import AbstractSampler
from .metropolis_hastings import *
from .._C_netket import sampler as c_sampler

from numba import jit, int64, float64
from .._jitclass import jitclass

@jitclass([("clusters", int64[:, :])])
class _exchange_kernel:
    def __init__(self, distances, d_max):
        clusters = []
        size = distances.shape[0]
        for i in range(size):
            for j in range(i + 1, size):
                if(distances[i][j] <= d_max):
                    clusters.append((i, j))

        self.clusters = _np.empty((len(clusters), 2), dtype=int64)

        for i, cluster in enumerate(clusters):
            self.clusters[i] = _np.asarray(cluster)

    def apply(self, state, state_1, log_prob_corr):

        clusters_size = self.clusters.shape[0]

        for k in range(state.shape[0]):
            state_1[k] = state[k]

            # pick a random cluster
            cl = _random.randint(0, clusters_size)

            # sites to be exchanged
            si = self.clusters[cl][0]
            sj = self.clusters[cl][1]

            state_1[k, si], state_1[k, sj] = state[k, sj], state[k, si]

        log_prob_corr[:] = 0.0


class MetropolisExchange(AbstractSampler):
    """
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
    """

    def __init__(self, machine, d_max=1, n_chains=16, sweep_size=None, batch_size=None):
        """
        Args:
              machine: A machine :math:`\Psi(s)` used for the sampling.
                       The probability distribution being sampled
                       from is :math:`F(\Psi(s))`, where the function
                       :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.

              d_max: The maximum graph distance allowed for exchanges.
              n_chains: The number of Markov Chain to be run in parallel on a single process.
              sweep_size: The number of exchanges that compose a single sweep.
                          If None, sweep_size is equal to the number of degrees of freedom (n_visible).
              batch_size: The batch size to be used when calling log_val on the given Machine.
                          If None, batch_size is equal to the number Markov chains (n_chains).


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
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisExchange(
                machine=machine,
                n_chains=n_chains,
                d_max=d_max,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            self.sampler = PyMetropolisHastings(
                machine,
                _exchange_kernel(
                    _np.asarray(machine.hilbert.graph.distances), d_max
                ),
                n_chains,
                sweep_size,
                batch_size,
            )
        super().__init__(machine, n_chains)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_pow(self):
        return self.sampler.machine_pow

    @machine_pow.setter
    def machine_pow(self, m_pow):
        self.sampler.machine_pow = m_pow

    @property
    def acceptance(self):
        """The measured acceptance probability."""
        return self.sampler.acceptance


class MetropolisExchangePt(AbstractSampler):
    """
    This sampler performs parallel-tempering
    moves in addition to the local moves implemented in `MetropolisExchange`.
    The number of replicas can be chosen by the user.
    """

    def __init__(
        self, machine, d_max=1, n_replicas=16, sweep_size=None, batch_size=None
    ):
        """
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
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisExchangePt(
                machine=machine,
                n_replicas=n_replicas,
                d_max=d_max,
                sweep_size=sweep_size,
            )
        else:
            raise ValueError(
                """Parallel Tempering samplers are not yet implemented
                for pure python machines"""
            )
        super().__init__(machine, 1)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_pow(self):
        return self.sampler.machine_pow

    @machine_pow.setter
    def machine_pow(self, m_pow):
        self.sampler.machine_pow = m_pow

    @property
    def acceptance(self):
        """The measured acceptance probability."""
        return self.sampler.acceptance
