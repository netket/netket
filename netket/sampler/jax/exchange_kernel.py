import jax
import numpy


class _ExchangeKernel:
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
