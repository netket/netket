import numpy as _np
from .abstract_sampler import AbstractSampler


class PyMetropolisLocal(AbstractSampler):
    """
    Pure Python implementation of local Metropolis Sampling.
    """

    def __init__(self, machine, n_chains, sweep_size=None):
        self.machine = machine
        self._n_visible = machine.hilbert.size
        self.n_chains = n_chains
        self._local_states = machine.hilbert.local_states
        self._state = _np.zeros((n_chains, self._n_visible))
        self._state1 = _np.zeros(self._state.shape)
        self._log_values = _np.zeros(n_chains, dtype=_np.complex128)
        self._log_values_1 = _np.zeros(n_chains, dtype=_np.complex128)
        self.machine_func = lambda x: _np.square(_np.absolute(x))

        self.sweep_size = sweep_size if sweep_size != None else self._n_visible
        self.reset(True)
        super().__init__(machine, self._state.shape)

    def reset(self, init_random=False):
        if init_random:
            self._state = _np.random.choice(
                self._local_states, size=self._state.size
            ).reshape(self._state.shape)
        self._log_values = self.machine.log_val(self._state)

    def propose(self):
        self._state1 = _np.copy(self._state)

        mask = _np.random.randint(0, self._n_visible, size=(self.n_chains, 1))
        values = _np.random.choice(self._local_states, size=self.n_chains).reshape(
            -1, 1
        )
        _np.put_along_axis(self._state1, mask, values, axis=1)

    def __next__(self):
        for _ in range(self.sweep_size):
            self.propose()

            self._log_values_1 = self.machine.log_val(self._state1)
            prob = self.machine_func(_np.exp(self._log_values_1 - self._log_values))

            accept = prob > _np.random.rand(self.n_chains)
            self._log_values = _np.where(accept, self._log_values_1, self._log_values)
            self._state = _np.where(accept.reshape(-1, 1), self._state1, self._state)
        return self._state
