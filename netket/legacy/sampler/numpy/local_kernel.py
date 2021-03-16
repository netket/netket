import numpy as _np
from netket.legacy import random as _random
from numba import jit, int64, float64
from numba.experimental import jitclass


@jitclass([("local_states", float64[:]), ("size", int64), ("n_states", int64)])
class _LocalKernel:
    def __init__(self, local_states, size):
        self.local_states = _np.sort(_np.asarray(local_states, dtype=_np.float64))
        self.size = size
        self.n_states = self.local_states.size

    def transition(self, state, state_1, log_prob_corr):

        for i in range(state.shape[0]):
            state_1[i] = state[i]

            si = _random.randint(0, self.size, size=())
            rs = _random.randint(0, self.n_states - 1, size=())

            state_1[i, si] = self.local_states[
                rs + (self.local_states[rs] >= state[i, si])
            ]

        log_prob_corr.fill(0.0)

    def random_state(self, state):

        for i in range(state.shape[0]):
            for si in range(state.shape[1]):
                rs = _random.randint(0, self.n_states, size=())
                state[i, si] = self.local_states[rs]
