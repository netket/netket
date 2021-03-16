from numba import jit
import numpy as _np
from netket.legacy import random as _random


class _CustomKernel:
    def __init__(self, move_operators, move_weights=None):

        self._rand_op_n = _np.empty(1, dtype=_np.intp)
        self._sections = _np.empty(1, dtype=_np.intp)
        self._x_prime = _np.empty(1)
        self._mels = _np.empty(1)
        self._get_conn = move_operators.get_conn_filtered
        self._n_operators = move_operators.n_operators

        if move_weights is None:
            self._move_weights = _np.ones(self._n_operators, dtype=_np.float64)
        else:
            self._move_weights = _np.asarray(move_weights, dtype=_np.float64)

        self._check_operators(move_operators.operators)

        # Check move weights
        if self._move_weights.shape != (self._n_operators,):
            raise ValueError("move_weights have the wrong shape")
        if self._move_weights.min() < 0:
            raise ValueError("move_weights must be positive")

        # normalize the probabilities and compute the cumulative
        self._move_weights /= self._move_weights.sum()
        self._move_cumulative = _np.cumsum(self._move_weights)

        self._hilbert = move_operators.hilbert

    def _check_operators(self, operators):
        for op in operators:
            assert op.imag.max() < 1.0e-10
            assert op.min() >= 0
            assert _np.allclose(op.sum(axis=0), 1.0)
            assert _np.allclose(op.sum(axis=1), 1.0)
            assert _np.allclose(op, op.T)

    def transition(self, state, state_1, log_prob_corr):

        self._rand_op_n, self._sections = self._pick_random_and_init(
            state.shape[0], self._move_cumulative, self._rand_op_n, self._sections
        )

        self._x_prime, self._mels = self._get_conn(
            state, self._sections, self._rand_op_n
        )

        self._choose_and_return(
            state_1, self._x_prime, self._mels, self._sections, log_prob_corr
        )

    @staticmethod
    @jit(nopython=True)
    def _pick_random_and_init(batch_size, move_cumulative, out, sections):

        if out.size != batch_size:
            out = _np.empty(batch_size, dtype=out.dtype)
            sections = _np.empty(batch_size, dtype=out.dtype)

        for i in range(batch_size):
            p = _random.uniform()
            out[i] = _np.searchsorted(move_cumulative, p)
        return out, sections

    @staticmethod
    @jit(nopython=True)
    def _choose_and_return(state_1, x_prime, mels, sections, log_prob_corr):
        low = 0
        for i in range(state_1.shape[0]):
            p = _random.uniform()
            exit_state = 0
            cumulative_prob = mels[low].real
            while p > cumulative_prob:
                exit_state += 1
                cumulative_prob += mels[low + exit_state].real
            state_1[i] = x_prime[low + exit_state]
            low = sections[i]

        log_prob_corr.fill(0.0)

    def random_state(self, state):

        for i in range(state.shape[0]):
            self._hilbert.random_state(out=state[i])
