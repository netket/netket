import numpy as _np
from .abstract_sampler import AbstractSampler

from .._C_netket import sampler as c_sampler
from .._C_netket.utils import random_engine
from ..stats import mean as _mean

from netket import random as _random

from numba import jit, int64, float64
from .._jitclass import jitclass


class PyMetropolisHastings(AbstractSampler):
    """
    ``MetropolisHastings`` is a generic Metropolis-Hastings sampler using
    a local transition kernel to perform moves in the Markov Chain.
    The transition kernel is used to generate
    a proposed state :math:`s^\prime`, starting from the current state :math:`s`.
    The move is accepted with probability

    .. math::
    A(s\rightarrow s^\prime) = \mathrm{min}\left (1,\frac{P(s^\prime)}{P(s)} F(e^{L(s,s^\prime)})\right),

    where the probability being sampled is :math:`F(\Psi(s))` (by default :math:`F(x)=|x|^2`)
    and :math:`L(s,s^\prime)` is a correcting factor computed by the transition kernel.
    """

    def __init__(
        self, machine, transition_kernel, n_chains=16, sweep_size=None, batch_size=None
    ):
        """
        Constructs a new ``MetropolisHastings`` sampler given a machine and
        a transition kernel.

        Args:
            machine: A machine :math:`\Psi(s)` used for the sampling.
                          The probability distribution being sampled
                          from is :math:`F(\Psi(s))`, where the function
                          $$F(X)$$, is arbitrary, by default :math:`F(X)=|X|^2`.
            transition_kernel: A function to generate a transition.
                          This should take as an input the current state (in batches)
                          and return a modified state (also in batches).
                          This function must also return an array containing the
                          `log_prob_corrections` :math:`L(s,s^\prime)`.
            n_chains: The number of Markov Chain to be run in parallel on a single process.
            sweep_size: The number of exchanges that compose a single sweep.
                        If None, sweep_size is equal to the number of degrees of freedom (n_visible).
            batch_size: The batch size to be used when calling log_val on the given Machine.
                        If None, batch_size is equal to the number Markov chains (n_chains).

        """

        self.machine = machine
        self.n_chains = n_chains

        self.sweep_size = sweep_size

        self._kernel = transition_kernel

        self.machine_pow = 2.0

        super().__init__(machine, n_chains)

    @property
    def n_chains(self):
        return self._n_chains

    @n_chains.setter
    def n_chains(self, n_chains):
        if n_chains < 0:
            raise ValueError("Expected a positive integer for n_chains ")

        self._n_chains = n_chains

        self._state = _np.zeros((n_chains, self._n_visible))
        self._state1 = _np.copy(self._state)

        self._log_values = _np.zeros(n_chains, dtype=_np.complex128)
        self._log_values_1 = _np.zeros(n_chains, dtype=_np.complex128)
        self._log_prob_corr = _np.zeros(n_chains)

    @property
    def machine_pow(self):
        return self._machine_pow

    @machine_pow.setter
    def machine_pow(self, m_power):
        self._machine_pow = m_power

    @property
    def sweep_size(self):
        return self._sweep_size

    @sweep_size.setter
    def sweep_size(self, sweep_size):
        self._sweep_size = sweep_size if sweep_size != None else self._n_visible
        if self._sweep_size < 0:
            raise ValueError("Expected a positive integer for sweep_size ")

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, machine):
        self._machine = machine
        self._n_visible = machine.hilbert.size
        self._hilbert = machine.hilbert

    def reset(self, init_random=False):
        if init_random:
            for state in self._state:
                self._hilbert.random_vals(state, random_engine())
        self.machine.log_val(self._state, out=self._log_values)

        self._accepted_samples = 0
        self._total_samples = 0

    @staticmethod
    @jit(nopython=True)
    def acceptance_kernel(
        state, state1, log_values, log_values_1, log_prob_corr, machine_pow
    ):
        accepted = 0

        for i in range(state.shape[0]):
            prob = _np.exp(
                machine_pow *
                (log_values_1[i] - log_values[i] + log_prob_corr[i]).real
            )

            if prob > _random.uniform(0, 1):
                log_values[i] = log_values_1[i]
                state[i] = state1[i]
                accepted += 1

        return accepted

    def __next__(self):

        _log_val = self.machine.log_val
        _acc_kernel = self.acceptance_kernel
        _state = self._state
        _state1 = self._state1
        _log_values = self._log_values
        _log_values_1 = self._log_values_1
        _log_prob_corr = self._log_prob_corr
        _machine_pow = self._machine_pow
        _accepted_samples = self._accepted_samples
        _t_kernel = self._kernel.apply

        for sweep in range(self.sweep_size):

            # Propose a new state using the transition kernel
            _t_kernel(_state, _state1, _log_prob_corr)

            _log_val(_state1, out=_log_values_1)

            # Acceptance Kernel
            acc = _acc_kernel(
                _state,
                _state1,
                _log_values,
                _log_values_1,
                _log_prob_corr,
                _machine_pow,
            )

            _accepted_samples += acc

        self._total_samples += self.sweep_size * self.n_chains

        return self._state

    @property
    def acceptance(self):
        """The measured acceptance probability."""
        return _mean(self._accepted_samples) / _mean(self._total_samples)
