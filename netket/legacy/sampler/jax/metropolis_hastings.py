from ..abstract_sampler import AbstractSampler

import jax
from functools import partial
from netket.legacy import random as _random


class MetropolisHastings(AbstractSampler):
    def __init__(self, machine, kernel, n_chains=16, sweep_size=None, rng_key=None):

        super().__init__(machine, n_chains)

        self._random_state_kernel = jax.jit(kernel.random_state)
        self._transition_kernel = jax.jit(kernel.transition)

        self._rng_key = rng_key
        if rng_key is None:
            self._rng_key = jax.random.PRNGKey(
                _random.randint(low=0, high=2 ** 32, size=()).item()
            )

        self.machine_pow = 2

        self.n_chains = n_chains

        self.sweep_size = sweep_size

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _metropolis_kernel(
        logpdf,
        transition_kernel,
        n_samples,
        sweep_size,
        initial_state,
        params,
        machine_pow,
        rng_key,
    ):
        # Array shapes are effectively static in jax-jitted code, so code will be
        # recompiled every time this changes.
        n_chains = initial_state.shape[0]

        def chains_one_step(i, walker):
            key, state, log_prob = walker

            # 1 to propagate for next iteration, 1 for uniform rng and n_chains for transition kernel
            keys = jax.random.split(key, 2 + n_chains)

            proposal = jax.vmap(transition_kernel, in_axes=(0, 0), out_axes=0)(
                keys[2:], state
            )
            proposal_log_prob = machine_pow * logpdf(params, proposal).real

            uniform = jax.random.uniform(keys[1], shape=(n_chains,))
            do_accept = uniform < jax.numpy.exp(proposal_log_prob - log_prob)

            # do_accept must match ndim of proposal and state (which is 2)
            state = jax.numpy.where(do_accept.reshape(-1, 1), proposal, state)

            log_prob = jax.numpy.where(
                do_accept.reshape(-1), proposal_log_prob, log_prob
            )

            return (keys[0], state, log_prob)

        # Loop over the sweeps
        def chains_one_sweep(walker, i):
            key, state, log_prob = walker
            walker = jax.lax.fori_loop(
                0, sweep_size, chains_one_step, (key, state, log_prob)
            )
            return walker, walker[1]

        keys = jax.random.split(rng_key, 2)
        initial_log_prob = machine_pow * logpdf(params, initial_state).real

        # Loop over the samples
        walker, samples = jax.lax.scan(
            chains_one_sweep,
            (keys[1], initial_state, initial_log_prob),
            xs=None,
            length=n_samples,
        )

        return keys[0], samples

    @property
    def n_chains(self):
        return self._n_chains

    @n_chains.setter
    def n_chains(self, n_chains):
        if n_chains < 0:
            raise ValueError("Expected a positive integer for n_chains ")

        self._n_chains = int(n_chains)
        self.reset(True)

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
        self._sweep_size = sweep_size if sweep_size != None else self._input_size
        if self._sweep_size < 0:
            raise ValueError("Expected a positive integer for sweep_size ")

    def reset(self, init_random=False):
        if init_random:

            self._rng_key, self._state = jax.lax.scan(
                self._random_state_kernel, self._rng_key, xs=None, length=self._n_chains
            )

            assert self._state.shape == self.sample_shape

        self._accepted_samples = 0
        self._total_samples = 0

    def generate_samples(self, n_samples, init_random=False, samples=None):
        if n_samples == 0:
            return

        self.reset(init_random)

        self._rng_key, samples = self._metropolis_kernel(
            self.machine._forward_fn_nj,
            self._transition_kernel,
            n_samples,
            self.sweep_size,
            self._state,
            self.machine.parameters,
            self.machine_pow,
            self._rng_key,
        )

        self._state = samples[-1]
        return samples

    def __next__(self):
        self._rng_key, samples = self._metropolis_kernel(
            self.machine._forward_fn_nj,
            self._transition_kernel,
            1,
            self.sweep_size,
            self._state,
            self.machine.parameters,
            self.machine_pow,
            self._rng_key,
        )

        self._state = samples[-1]
        return self._state
