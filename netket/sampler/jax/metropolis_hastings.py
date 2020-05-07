from ..abstract_sampler import AbstractSampler

import jax
from functools import partial
from netket import random as _random


class MetropolisHastings(AbstractSampler):
    def __init__(self, machine, kernel, n_chains=16, sweep_size=None, rng_key=None):

        super().__init__(machine, n_chains)

        self._random_state_kernel = jax.jit(kernel.random_state)
        self._transition_kernel = jax.jit(kernel.transition)

        self._rng_key = rng_key
        if rng_key is None:
            self._rng_key = jax.random.PRNGKey(_random.randint(low=0, high=(2 ** 32)))

        self.machine_pow = 2

        self.n_chains = n_chains

        self.sweep_size = sweep_size

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 2, 3, 5))
    def _metropolis_kernel(
        n_samples,
        sweep_size,
        n_chains,
        transition_kernel,
        initial_state,
        logpdf,
        params,
        machine_pow,
        rng_key,
    ):
        @jax.jit
        def one_step(i, walker):
            key, state, log_prob = walker

            keys = jax.random.split(key, 3)

            proposal = transition_kernel(keys[1], state)

            proposal_log_prob = machine_pow * logpdf(params, proposal).real

            uniform = jax.random.uniform(keys[2])
            do_accept = uniform < jax.numpy.exp(proposal_log_prob - log_prob)

            state = jax.np.where(do_accept, proposal, state)
            log_prob = jax.np.where(do_accept, proposal_log_prob, log_prob)

            return (keys[0], state, log_prob)

        @jax.jit
        def one_sweep(walker, i):
            key, state, log_prob = walker
            walker = jax.lax.fori_loop(0, sweep_size, one_step, (key, state, log_prob))
            return walker, walker[1]

        @jax.jit
        def single_chain_kernel(key, state):
            log_prob = machine_pow * logpdf(params, state).real
            walker, samples = jax.lax.scan(
                one_sweep, (key, state, log_prob), xs=None, length=n_samples
            )
            return samples

        run_mcmc = jax.vmap(single_chain_kernel, in_axes=(0, 0), out_axes=1)

        keys = jax.random.split(rng_key, n_chains + 1)
        samples = run_mcmc(keys[1:], initial_state)

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

            assert self._state.shape == (self._n_chains, self._input_size)

        self._accepted_samples = 0
        self._total_samples = 0

    def generate_samples(self, n_samples, init_random=False, samples=None):
        if n_samples == 0:
            return

        self.reset(init_random)

        self._rng_key, samples = self._metropolis_kernel(
            n_samples,
            self.sweep_size,
            self.n_chains,
            self._transition_kernel,
            self._state,
            self.machine.jax_forward,
            self.machine.parameters,
            self.machine_pow,
            self._rng_key,
        )

        self._state = samples[-1]
        return samples

    def __next__(self):
        self._rng_key, samples = self._metropolis_kernel(
            1,
            self.sweep_size,
            self.n_chains,
            self._transition_kernel,
            self._state,
            self.machine.jax_forward,
            self.machine.parameters,
            self.machine_pow,
            self._rng_key,
        )

        self._state = samples[-1]
        return self._state
