# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from functools import partial

from typing import Any, Tuple, Callable

import numpy as np
from numba import jit
from jax import numpy as jnp
import jax

from netket.hilbert import AbstractHilbert
from netket.utils.mpi import mpi_sum, n_nodes
from netket.utils.types import PyTree
from netket.utils.deprecation import deprecated

import netket.jax as nkjax

from .metropolis import MetropolisSampler


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

    n_steps_proc: int = 0
    """Number of moves performed along the chains in this process since the last reset."""
    n_accepted_proc: int = 0
    """Number of accepted transitions among the chains in this process since the last reset."""

    @property
    def acceptance(self) -> float:
        """The fraction of accepted moves across all chains and MPI processes.

        The rate is computed since the last reset of the sampler.
        Will return None if no sampling has been performed since then.
        """
        if self.n_steps == 0:
            return None

        return self.n_accepted / self.n_steps

    @property
    @deprecated(
        """Please use the attribute `.acceptance` instead of
        `.acceptance_ratio`. The new attribute `.acceptance` returns the
        acceptance ratio ∈ [0,1], instead of the current `acceptance_ratio`
        returning a percentage, which is a bug."""
    )
    def acceptance_ratio(self) -> float:
        """DEPRECATED: Please use the attribute `.acceptance` instead of
        `.acceptance_ratio`. The new attribute `.acceptance` returns the
        acceptance ratio ∈ [0,1], instead of the current `acceptance_ratio`
        returning a percentage, which is a bug.

        The percentage of accepted moves across all chains and MPI processes.

        The rate is computed since the last reset of the sampler.
        Will return None if no sampling has been performed since then.
        """
        return self.acceptance * 100

    @property
    def n_steps(self) -> int:
        """Total number of moves performed across all processes since the last reset."""
        return self.n_steps_proc * n_nodes

    @property
    def n_accepted(self) -> int:
        """Total number of moves accepted across all processes since the last reset."""
        return mpi_sum(self.n_accepted_proc)

    def __repr__(self):
        if self.n_steps > 0:
            acc_string = "# accepted = {}/{} ({}%), ".format(
                self.n_accepted, self.n_steps, self.acceptance * 100
            )
        else:
            acc_string = ""

        return f"MetropolisNumpySamplerState({acc_string}rng state={self.rng})"


@partial(jax.jit, static_argnums=0)
def apply_model(machine, pars, weights):
    return machine.apply(pars, weights)


class MetropolisSamplerNumpy(MetropolisSampler):
    """
    Metropolis-Hastings sampler for an Hilbert space according to a specific transition
    rule executed on CPU through Numpy.

    This sampler is equivalent to :ref:`netket.sampler.MetropolisSampler` but instead of
    executing the whole sampling inside a jax-jitted function, only evaluates the forward
    pass inside a jax-jitted function, while proposing new steps and accepting/rejecting
    them is performed in numpy.

    Because of Jax dispatch cost, and especially for small system, this sampler performs
    poorly, while asyntotically it should have the same performance of standard Jax samplers.

    However, some transition rules don't work on GPU, and some samplers (Hamiltonian) work
    very poorly on jax so this is a good workaround.

    See :ref:`netket.sampler.MetropolisSampler` for more informations.
    """

    def _init_state(sampler, machine, parameters, key):
        rgen = np.random.default_rng(np.asarray(key))

        σ = np.zeros((sampler.n_batches, sampler.hilbert.size), dtype=sampler.dtype)

        ma_out = jax.eval_shape(machine.apply, parameters, σ)

        state = MetropolisNumpySamplerState(
            σ=σ,
            σ1=np.copy(σ),
            log_values=np.zeros(sampler.n_batches, dtype=ma_out.dtype),
            log_values_1=np.zeros(sampler.n_batches, dtype=ma_out.dtype),
            log_prob_corr=np.zeros(
                sampler.n_batches, dtype=nkjax.dtype_real(ma_out.dtype)
            ),
            rng=rgen,
            rule_state=sampler.rule.init_state(sampler, machine, parameters, rgen),
        )

        if not sampler.reset_chains:
            key = jnp.asarray(
                state.rng.integers(0, 1 << 32, size=2, dtype=np.uint32), dtype=np.uint32
            )

            state.σ = np.copy(
                sampler.rule.random_state(sampler, machine, parameters, state, key)
            )

        return state

    def _reset(sampler, machine, parameters, state):
        if sampler.reset_chains:
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

        state.n_steps_proc += sampler.n_sweeps * sampler.n_chains
        state.n_accepted_proc += accepted

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

    def __repr__(sampler):
        return (
            "MetropolisSamplerNumpy("
            + "\n  hilbert = {},".format(sampler.hilbert)
            + "\n  rule = {},".format(sampler.rule)
            + "\n  n_chains = {},".format(sampler.n_chains)
            + "\n  machine_power = {},".format(sampler.machine_pow)
            + "\n  reset_chains = {},".format(sampler.reset_chains)
            + "\n  n_sweeps = {},".format(sampler.n_sweeps)
            + "\n  dtype = {},".format(sampler.dtype)
            + ")"
        )

    def __str__(sampler):
        return (
            "MetropolisSamplerNumpy("
            + "rule = {}, ".format(sampler.rule)
            + "n_chains = {}, ".format(sampler.n_chains)
            + "machine_power = {}, ".format(sampler.machine_pow)
            + "n_sweeps = {}, ".format(sampler.n_sweeps)
            + "dtype = {})".format(sampler.dtype)
        )


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


def MetropolisLocalNumpy(hilbert: AbstractHilbert, *args, **kwargs):
    from .rules import LocalRuleNumpy

    rule = LocalRuleNumpy()
    return MetropolisSamplerNumpy(hilbert, rule, *args, **kwargs)


def MetropolisHamiltonianNumpy(hilbert: AbstractHilbert, hamiltonian, *args, **kwargs):
    from .rules import HamiltonianRuleNumpy

    rule = HamiltonianRuleNumpy(hamiltonian)
    return MetropolisSamplerNumpy(hilbert, rule, *args, **kwargs)


def MetropolisCustomNumpy(
    hilbert: AbstractHilbert, move_operators, move_weights=None, *args, **kwargs
):
    from .rules import CustomRuleNumpy

    rule = CustomRuleNumpy(move_operators, move_weights)
    return MetropolisSamplerNumpy(hilbert, rule, *args, **kwargs)
