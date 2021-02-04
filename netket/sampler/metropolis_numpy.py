# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from functools import partial

from typing import Optional, Any, Union, Tuple, Callable

import numpy as np
from numba import jit, int64, float64
from jax import numpy as jnp
import jax

from netket.hilbert import AbstractHilbert

from .metropolis import MetropolisSampler

PyTree = Any
PRNGKeyType = jnp.ndarray
SeedType = Union[int, PRNGKeyType]


@dataclass
class MetropolisNumpySamplerState:
    σ: np.ndarray
    """Holds the current configuration."""
    σ1: np.ndarray
    """Holds a proposed configuration (preallocation)."""

    log_values: np.ndarray
    """Holds model(pars, σ) for the current σ (preallocation)."""
    log_values_1: np.ndarray
    """Holds model(pars, σ1) for the last σ1 (preallocation)."""
    log_prob_corr: np.ndarray
    """Holds optional acceptance correction (preallocation)."""

    rule_state: Any
    """The optional state of the rule."""
    rng: Any
    """A numpy random generator."""

    n_samples: int = 0
    """Number of moves performed along the chains since the last reset."""
    n_accepted: int = 0
    """Number of accepted transitions along the chains since the last reset."""


@partial(jax.jit, static_argnums=0)
def apply_model(machine, pars, weights):
    return machine(pars, weights)


class MetropolisSamplerNumpy(MetropolisSampler):
    def _init_state(sampler, machine, parameters, key):
        rgen = np.random.default_rng(np.asarray(key))

        σ = np.zeros((sampler.n_batches, sampler.hilbert.size), dtype=sampler.dtype)

        ma_out = jax.eval_shape(machine, parameters, σ)

        state = MetropolisNumpySamplerState(
            σ=σ,
            σ1=np.copy(σ),
            log_values=np.zeros(sampler.n_batches, dtype=ma_out.dtype),
            log_values_1=np.zeros(sampler.n_batches, dtype=ma_out.dtype),
            log_prob_corr=np.zeros(
                sampler.n_batches, dtype=jax.dtypes.dtype_real(ma_out.dtype)
            ),
            rng=rgen,
            rule_state=sampler.rule.init_state(sampler, machine, parameters, rgen),
        )

        if sampler.reset_chain:
            key = jnp.asarray(
                state.rng.integers(0, 1 << 32, size=2, dtype=np.uint32), dtype=np.uint32
            )

            state.σ = np.copy(
                sampler.rule.random_state(sampler, machine, parameters, state, key)
            )

        return state

    def _reset(sampler, machine, parameters, state):
        if sampler.reset_chain:
            # directly generate a PRNGKey which is a [2xuint32] array
            key = jnp.asarray(
                state.rng.integers(0, 1 << 32, size=2, dtype=np.uint32), dtype=np.uint32
            )
            state.σ = np.copy(
                sampler.rule.random_state(sampler, machine, parameters, state, key)
            )

        state.rule_state = sampler.rule.reset(sampler, machine, parameters, state)
        state.log_values = np.copy(apply_model(machine, parameters, state.σ))

        state._accepted_samples = 0
        state._total_samples = 0

        return state

    def _sample_next(sampler, machine, parameters, state):
        σ = state.σ
        σ1 = state.σ1
        log_values = state.log_values
        log_values_1 = state.log_values_1
        log_prob_corr = state.log_prob_corr
        mpow = sampler.machine_pow

        rgen = state.rng

        accepted = 0

        for sweep in range(sampler.n_sweeps):
            # Propose a new state using the transition kernel
            # σp, log_prob_correction =
            sampler.rule.transition(sampler, machine, parameters, state, state.rng, σ)

            log_values_1 = np.asarray(apply_model(machine, parameters, σ1))

            random_uniform = rgen.uniform(0, 1, size=σ.shape[0])

            # Acceptance Kernel
            accepted += acceptance_kernel(
                σ,
                σ1,
                log_values,
                log_values_1,
                log_prob_corr,
                mpow,
                random_uniform,
            )

        state.n_samples += sampler.n_sweeps * sampler.n_chains
        state.n_accepted += accepted

        return state, state.σ

    def _sample_chain(
        sampler,
        machine: Callable,
        parameters: PyTree,
        state: MetropolisNumpySamplerState,
        chain_length: int,
    ) -> Tuple[jnp.ndarray, MetropolisNumpySamplerState]:

        samples = np.empty(
            (chain_length, sampler.n_chains, sampler.hilbert.size), dtype=sampler.dtype
        )

        for i in range(chain_length):
            state, σ = sampler.sample_next(machine, parameters, state)
            samples[i] = σ

        return samples, state


@jit(nopython=True)
def acceptance_kernel(
    σ, σ1, log_values, log_values_1, log_prob_corr, machine_pow, random_uniform
):
    accepted = 0

    for i in range(σ.shape[0]):
        prob = np.exp(
            machine_pow * (log_values_1[i] - log_values[i]).real + log_prob_corr[i]
        )
        assert not math.isnan(prob)

        if prob > random_uniform[i]:
            log_values[i] = log_values_1[i]
            σ[i] = σ1[i]
            accepted += 1

    return accepted


# def MetropolisSamplerNumpy(
#     hilbert: AbstractHilbert,
#     rule,
#     n_sweeps: Optional[int] = None,
#     **kwargs,
# ):
#     if n_sweeps is None:
#         n_sweeps = hilbert.size

#     return MetropolisSamplerNumpy_(
#         hilbert=hilbert, rule=rule, n_sweeps=n_sweeps, **kwargs
#     )


from .rules import HamiltonianRuleNumpy, CustomRuleNumpy


def MetropolisHamiltonianNumpy(hilbert: AbstractHilbert, hamiltonian, *args, **kwargs):
    rule = HamiltonianRuleNumpy(hamiltonian)
    return MetropolisSamplerNumpy(hilbert, rule, *args, **kwargs)


def MetropolisCustomNumpy(
    hilbert: AbstractHilbert, move_operators, move_weights=None, *args, **kwargs
):
    rule = CustomRuleNumpy(move_operators, move_weights)
    return MetropolisSamplerNumpy(hilbert, rule, *args, **kwargs)
