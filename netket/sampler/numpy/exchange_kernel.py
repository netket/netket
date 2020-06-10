import numpy as _np
from netket import random as _random
from numba import jit, int64, float64


class _ExchangeKernel:
    def __init__(self, hilbert, d_max):
        clusters = []
        distances = _np.asarray(hilbert.graph.distances())
        size = distances.shape[0]
        for i in range(size):
            for j in range(i + 1, size):
                if distances[i][j] <= d_max:
                    clusters.append((i, j))

        self.clusters = _np.empty((len(clusters), 2), dtype=_np.int64)

        for i, cluster in enumerate(clusters):
            self.clusters[i] = _np.asarray(cluster)

        self._hilbert = hilbert

    @staticmethod
    @jit(nopython=True)
    def _transition(state, state_1, log_prob_corr, clusters):

        clusters_size = clusters.shape[0]

        for k in range(state.shape[0]):
            state_1[k] = state[k]

            # pick a random cluster
            cl = _random.randint(0, clusters_size)

            # sites to be exchanged
            si = clusters[cl][0]
            sj = clusters[cl][1]

            state_1[k, si], state_1[k, sj] = state[k, sj], state[k, si]

        log_prob_corr[:] = 0.0

    def transition(self, state, state_1, log_prob_corr):
        return self._transition(state, state_1, log_prob_corr, self.clusters)

    def random_state(self, state):

        for i in range(state.shape[0]):
            self._hilbert.random_vals(out=state[i])
