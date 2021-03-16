from numba import jit
import math

import numpy as _np
from netket.legacy import random as _random


class _HamiltonianKernel:
    def __init__(self, hamiltonian):
        self._hamiltonian = hamiltonian
        self._sections = _np.empty(1, dtype=_np.int32)
        self._hamconn = self._hamiltonian.get_conn_flattened
        self._n_conn = self._hamiltonian.n_conn
        self._hilbert = hamiltonian.hilbert

    def transition(self, state, state_1, log_prob_corr):

        sections = self._sections
        sections = _np.empty(state.shape[0], dtype=_np.int32)
        vprimes = self._hamconn(state, sections)[0]

        self._choose(vprimes, sections, state_1, log_prob_corr)

        self._n_conn(state_1, sections)

        log_prob_corr -= _np.log(sections)

    def random_state(self, state):

        for i in range(state.shape[0]):
            self._hilbert.random_state(out=state[i])

    @staticmethod
    @jit(nopython=True)
    def _choose(states, sections, out, w):
        low_range = 0
        for i, s in enumerate(sections):
            n_rand = _random.randint(low_range, s, size=())
            out[i] = states[n_rand]
            w[i] = math.log(s - low_range)
            low_range = s
