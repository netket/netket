from ._C_netket.sampler import *

import numpy as _np


def compute_samples(
    sampler,
    n_samples,
    n_discard=None,
    compute_log_values=False,
    samples=None,
    log_values=None,
):

    n_chains = sampler.n_chains
    n_samples = int(_np.ceil((n_samples / n_chains)))

    if samples is None or log_values is None:
        samples = _np.ndarray((n_samples, n_chains, sampler.machine.hilbert.size))
        if compute_log_values:
            log_values = _np.ndarray((n_samples, n_chains), dtype=_np.complex128)

    if n_discard is None:
        n_discard = n_samples // 10

    sweep = sampler.sweep

    # Burnout phase
    for _ in range(n_discard):
        sweep()

    if compute_log_values:
        state = sampler.current_state
        # Generate samples
        for i in range(n_samples):
            sweep()
            samples[i], log_values[i] = state

        return samples, log_values
    else:
        state = sampler.current_sample
        # Generate samples
        for i in range(n_samples):
            sweep()
            samples[i] = state

        return samples


class PyMetropolisLocal(object):
    """
    Metropolis Sampling.
    """

    def __init__(self, machine, n_chains, sweep_size=None):
        self.machine = machine
        self._local_states = machine.hilbert.local_states
        self._n_visible = machine.hilbert.size
        self.n_chains = n_chains
        self._state = _np.zeros((n_chains, self._n_visible))
        self._state1 = _np.zeros(self._state.shape)
        self._log_values = _np.zeros(n_chains, dtype=_np.complex128)
        self._log_values_1 = _np.zeros(n_chains, dtype=_np.complex128)
        self.machine_func = lambda x: _np.square(_np.absolute(x))

        self.sweep_size = sweep_size if sweep_size != None else self._n_visible
        self.reset(True)

    def reset(self, init_random=False):
        if init_random:
            self._state = _np.random.choice(
                self._local_states, size=self._state.size
            ).reshape(self._state.shape)
        self._log_values = self.machine.log_val(self._state)

    @property
    def current_sample(self):
        return self._state

    @property
    def current_state(self):
        return (self._state, self._log_values)

    def propose(self):
        self._state1 = _np.copy(self._state)

        mask = _np.random.randint(0, self._n_visible, size=(self.n_chains, 1))
        values = _np.random.choice(self._local_states, size=self.n_chains).reshape(
            -1, 1
        )
        _np.put_along_axis(self._state1, mask, values, axis=1)

    def sweep(self):
        for _ in range(self.sweep_size):
            self.propose()

            self._log_values_1 = self.machine.log_val(self._state1)
            prob = self.machine_func(_np.exp(self._log_values_1 - self._log_values))

            accept = prob > _np.random.rand(self.n_chains)
            self._log_values = _np.where(accept, self._log_values_1, self._log_values)
            self._state = _np.where(accept.reshape(-1, 1), self._state1, self._state)
