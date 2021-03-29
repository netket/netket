import jax
import numpy


class _JaxExchangeKernel:
    def __init__(self, hilbert, clusters):
        self.clusters = jax.numpy.asarray(clusters)

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
        return key, jax.numpy.asarray(self._hilbert.random_state())
