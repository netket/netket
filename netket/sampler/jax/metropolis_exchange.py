import jax
from .metropolis_hastings import MetropolisHastings
import numpy


class _exchange_kernel:
    def __init__(self, hilbert, d_max):
        clusters = []
        distances = jax.numpy.asarray(hilbert.graph.distances())
        size = distances.shape[0]
        for i in range(size):
            for j in range(i + 1, size):
                if distances[i][j] <= d_max:
                    clusters.append((i, j))

        self.clusters = numpy.empty((len(clusters), 2), dtype=numpy.int64)

        for i, cluster in enumerate(clusters):
            self.clusters[i] = numpy.asarray(cluster)
        self.clusters = jax.numpy.asarray(self.clusters)

        self.clusters_size = self.clusters.shape[0]

        self._hilbert = hilbert

    def transition(self, key, state):

        # pick a random cluster
        cl = jax.random.randint(key, shape=(1,), minval=0, maxval=self.clusters_size)

        # sites to be exchanged
        si = self.clusters[cl, 0]
        sj = self.clusters[cl, 1]

        state_1 = jax.ops.index_update(state, si, state[sj])
        return jax.ops.index_update(state_1, sj, state[si])

    def random_state(self, key, state):
        return key, jax.numpy.asarray(self._hilbert.random_vals())


class MetropolisExchange(MetropolisHastings):
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
    """

    def __init__(self, machine, d_max=1, n_chains=8, sweep_size=None):
        r"""
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
        super().__init__(
            machine, _exchange_kernel(machine.hilbert, d_max), n_chains, sweep_size,
        )
