import numpy as _np
from .abstract_sampler import AbstractSampler

from ..stats import mean as _mean

from numba import jit, jitclass
from numba import int64, float64
from netket import random as _random
import math


class PyMetropolisHastingsPt(AbstractSampler):
    """
    ``MetropolisHastingsPt`` is a generic Metropolis-Hastings sampler using
    a local transition kernel to perform moves in the Markov Chain and replica-exchange moves
    to increase mixing times.
    The transition kernel is used to generate a proposed state :math:`s^\prime`,
    starting from the current state :math:`s`, as in ``MetropolisHastings``.
    Each replica is at an inverse temperature :math:`\beta` linearly spaced in the interval [0,1],
    where :math:`beta=1` corresponds to the target probability distribution,
    and :math:`beta=0` is the flat distribution.
    """

    def __init__(
        self, machine, transition_kernel, n_replicas=32, sweep_size=None, batch_size=None
    ):
        """
        Constructs a new ``MetropolisHastingsPt`` sampler given a machine and
        a transition kernel and a number of replicas.

        Args:
            machine: A machine :math:`\Psi(s)` used for the sampling.
                          The probability distribution being sampled
                          from is :math:`|\Psi(s)|^p`, where the exponent p
                          is arbitrary and by default :math:`p=2`.
            transition_kernel: A function to generate a transition.
                          This should take as an input the current state (in batches)
                          and return a modified state (also in batches).
                          This function must also return an array containing the
                          `log_prob_corrections` :math:`L(s,s^\prime)`.
            n_replicas (int): The number of replicas used for replica-exchange moves. Each replica samples
            sweep_size (int): The number of exchanges that compose a single sweep.
                        If None, sweep_size is equal to the number of degrees of freedom (n_visible).
            batch_size (int): The batch size to be used when calling log_val on the given Machine.
                        If None, batch_size is equal to the number of replicas (n_replicas).

        """

        self.machine = machine
        self.n_replicas = n_replicas

        self.sweep_size = sweep_size

        self._kernel = transition_kernel

        self.machine_pow = 2.0

        # Contains quantities to compute diffusion coefficient of replicas
        # beta_stats[0] = position of replica at beta=1 (do not reset this!)
        # beta_stats[1] = running average of beta=1 position
        # beta_stats[2] = running average of beta=1 position**2
        # beta_stats[3] = current number of exchange steps performed
        self._beta_stats = _np.zeros(4)

        super().__init__(machine)

    @property
    def n_replicas(self):
        return self._n_replicas

    @n_replicas.setter
    def n_replicas(self, n_replicas):
        if n_replicas < 1:
            raise ValueError("Expected n_replicas>0. ")

        self._n_replicas = n_replicas

        self._state = _np.zeros((n_replicas, self._n_visible))
        self._state1 = _np.copy(self._state)

        self._log_values = _np.zeros(n_replicas, dtype=_np.complex128)
        self._log_values_1 = _np.zeros(n_replicas, dtype=_np.complex128)
        self._log_prob_corr = _np.zeros(n_replicas)

        # Linearly spaced inverse temperature
        self._beta = _np.empty(n_replicas)

        for i in range(n_replicas):
            self._beta[i] = (1. - float(i) / float(n_replicas))

        # some temporary arrays
        self._proposed_beta = _np.empty(n_replicas)
        self._beta_prob = _np.empty(n_replicas)

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
                self._hilbert.random_vals(out=state)
        self._log_values = self.machine.log_val(
            self._state, out=self._log_values)

        self._accepted_samples = _np.zeros(self._n_replicas)
        self._total_samples = 0

        self._beta_stats[1:].fill(0)

    @staticmethod
    @jit(nopython=True)
    def _fixed_beta_acceptance_kernel(
        state, state1, log_values, log_values_1,
        log_prob_corr, machine_pow, beta, accepted
    ):

        for i in range(state.shape[0]):
            prob = math.exp(
                machine_pow * beta[i] *
                (log_values_1[i] - log_values[i] + log_prob_corr[i]).real
            )

            assert(not math.isnan(prob))

            accept = prob > _random.uniform(0, 1)

            if (accept):
                log_values[i] = log_values_1[i]
                state[i] = state1[i]
                accepted[i] += 1

    def __next__(self):

        _log_val = self.machine.log_val
        _fb_acc_kernel = self._fixed_beta_acceptance_kernel
        _state = self._state
        _state1 = self._state1
        _log_values = self._log_values
        _log_values_1 = self._log_values_1
        _log_prob_corr = self._log_prob_corr
        _machine_pow = self._machine_pow
        _accepted_samples = self._accepted_samples
        _t_kernel = self._kernel.apply
        _beta = self._beta
        _beta_stats = self._beta_stats
        _proposed_beta = self._proposed_beta
        _beta_prob = self._beta_prob

        for sweep in range(self.sweep_size):

            # Propose a new state using the transition kernel
            _t_kernel(_state, _state1, _log_prob_corr)

            _log_values_1 = _log_val(_state1, out=_log_values_1)

            # Acceptance Kernel for fixed-beta moves
            _fb_acc_kernel(
                _state,
                _state1,
                _log_values,
                _log_values_1,
                _log_prob_corr,
                _machine_pow,
                _beta,
                _accepted_samples
            )

            # Transition + Acceptance Kernel for replica exchange moves
            self._exchange_step_kernel(
                _log_values,
                _machine_pow,
                _beta,
                _proposed_beta,
                _beta_prob,
                _beta_stats,
                _accepted_samples)

        self._total_samples += self.sweep_size
        return self._state[_np.intp(_beta_stats[0])].reshape(1, -1)

    @staticmethod
    @jit(nopython=True)
    def _exchange_step_kernel(log_values,
                              machine_pow,
                              beta,
                              proposed_beta,
                              prob,
                              beta_stats,
                              accepted_samples):
        # Choose a random swap order (odd/even swap)
        swap_order = _random.randint(0, 2)

        n_replicas = beta.shape[0]

        for i in range(swap_order, n_replicas, 2):
            inn = (i + 1) % n_replicas
            proposed_beta[i] = beta[inn]
            proposed_beta[inn] = beta[i]

        for i in range(n_replicas):
            prob[i] = math.exp(
                machine_pow * (proposed_beta[i] - beta[i]) * log_values[i].real)

        for i in range(swap_order, n_replicas, 2):
            inn = (i + 1) % n_replicas

            prob[i] *= prob[inn]

            if(prob[i] > _random.uniform(0, 1)):
                # swapping status
                beta[i], beta[inn] = beta[inn], beta[i]
                accepted_samples[i], accepted_samples[inn] = accepted_samples[inn], accepted_samples[i]

                if(beta_stats[0] == i):
                    beta_stats[0] = inn
                elif(beta_stats[0] == inn):
                    beta_stats[0] = i

        # Update statistics to compute diffusion coefficient of replicas
        # Total exchange steps performed
        beta_stats[-1] += 1

        delta = beta_stats[0] - beta_stats[1]
        beta_stats[1] += delta / float(beta_stats[-1])
        delta2 = beta_stats[0] - beta_stats[1]
        beta_stats[2] += delta * delta2

    @property
    def stats(self):
        stats = {}
        accept = self._accepted_samples / float(self._total_samples)

        stats["mean_acceptance"] = _mean(accept)
        stats["min_acceptance"] = _mean(accept.min())
        stats["max_acceptance"] = _mean(accept.max())

        # Average position of beta=1
        # This is normalized and centered around zero
        # In the ideal case the average should be zero
        stats["normalized_beta=1_position"] = self._beta_stats[1] / \
            float(self._n_replicas - 1) - 0.5

        # Average variance on the position of beta=1
        # In the ideal case this quantity should be of order ~ [0.2, 1]
        stats["normalized_beta=1_diffusion"] = _np.sqrt(
            self._beta_stats[2] / self._beta_stats[-1]) / float(self._n_replicas)

        return stats
