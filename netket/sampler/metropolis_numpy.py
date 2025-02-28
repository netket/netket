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
from functools import partial, wraps

from typing import Any
from collections.abc import Callable

import numpy as np
from numba import jit
from jax import numpy as jnp
import jax

from netket.hilbert import AbstractHilbert
from netket.utils.mpi import mpi_sum, n_nodes
from netket.utils.types import PyTree
from netket import config

import netket.jax as nkjax

from .metropolis import MetropolisSampler, MetropolisRule


@dataclass
class MetropolisNumpySamplerState:
    σ: np.ndarray
    """Holds the current configuration."""
    σ1: np.ndarray
    """Holds a proposed configuration (preallocation)."""

    log_prob: np.ndarray
    """Holds model(pars, σ) for the current σ (preallocation)."""
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
    def n_steps(self) -> int:
        """Total number of moves performed across all processes since the last reset."""
        return self.n_steps_proc * n_nodes

    @property
    def n_accepted(self) -> int:
        """Total number of moves accepted across all processes since the last reset."""
        return mpi_sum(self.n_accepted_proc)

    def __repr__(self):
        if self.n_steps > 0:
            acc_string = f"# accepted = {self.n_accepted}/{self.n_steps} ({self.acceptance * 100}%), "
        else:
            acc_string = ""

        return f"MetropolisNumpySamplerState({acc_string}rng state={self.rng})"


@partial(jax.jit, static_argnums=(0, 3))
def apply_model(machine, pars, weights, chunk_size):
    chunked = nkjax.apply_chunked(
        machine.apply, in_axes=(None, 0), chunk_size=chunk_size
    )
    return chunked(pars, weights)


class MetropolisSamplerNumpy(MetropolisSampler):
    """
    Metropolis-Hastings sampler for an Hilbert space according to a specific transition
    rule executed on CPU through Numpy.

    This sampler is equivalent to :ref:`netket.sampler.MetropolisSampler` but instead of
    executing the whole sampling inside a jax-jitted function, only evaluates the forward
    pass inside a jax-jitted function, while proposing new steps and accepting/rejecting
    them is performed in numpy.

    Because of Jax dispatch cost, and especially for small system, this sampler performs
    poorly, while asymptotically it should have the same performance of standard Jax samplers.

    However, some transition rules don't work on GPU, and some samplers (Hamiltonian) work
    very poorly on jax so this is a good workaround.

    See :ref:`netket.sampler.MetropolisSampler` for more information.

    .. warning::
       This sampler only works on the CPU. To use the Hamiltonian sampler with GPUs, you
       should use :class:`netket.sampler.MetropolisHamiltonian`.

       This sampler is largely outdated, and we recommend not to use them. Little effort has
       been put into improving performance in those samplers after 2021.

       If you have a complicated transitions rule, consider instead using a `jax.pure_callback`.
    """

    @wraps(MetropolisSampler.__init__)
    def __init__(self, hilbert: AbstractHilbert, rule: MetropolisRule, **kwargs):
        super().__init__(hilbert, rule, **kwargs)
        # standard samplers use jax arrays, this must be a numpy array
        self.machine_pow = np.array(self.machine_pow)

    @property
    def n_batches(self) -> int:
        r"""
        The batch size of the configuration $\sigma$ used by this sampler on this
        jax process.

        This is used to determine the shape of the batches generated in a single process.
        This is needed because when using MPI, every process must create a batch of chains
        of :attr:`~Sampler.n_chains_per_rank`, while when using the experimental sharding
        mode we must declare the full shape on every jax process, therefore this returns
        :attr:`~Sampler.n_chains`.

        Usage of this flag is required to support both MPI and sharding.

        Samplers may override this to have a larger batch size, for example to
        propagate multiple replicas (in the case of parallel tempering).
        """
        return self.n_chains_per_rank

    def _init_state(sampler, machine, parameters, key):
        rgen = np.random.default_rng(np.asarray(key))

        σ = np.zeros((sampler.n_batches, sampler.hilbert.size), dtype=sampler.dtype)

        ma_out = jax.eval_shape(machine.apply, parameters, σ)

        state = MetropolisNumpySamplerState(
            σ=σ,
            σ1=np.copy(σ),
            log_prob=np.zeros(sampler.n_batches, dtype=ma_out.dtype),
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
        state.log_prob = np.array(
            sampler.machine_pow
            * apply_model(machine, parameters, state.σ, sampler.chunk_size).real
        )

        state._accepted_samples = 0
        state._total_samples = 0

        return state

    def _sample_next(sampler, machine, parameters, state):
        σ = state.σ
        σ1 = state.σ1
        log_prob = state.log_prob
        log_prob_corr = state.log_prob_corr
        mpow = sampler.machine_pow

        rgen = state.rng

        accepted = 0

        for sweep in range(sampler.sweep_size):
            # Propose a new state using the transition kernel
            # σp, log_prob_correction =
            sampler.rule.transition(sampler, machine, parameters, state, state.rng, σ)

            if config.netket_experimental_sharding:
                from jax.experimental.multihost_utils import (
                    host_local_array_to_global_array,
                    global_array_to_host_local_array,
                )

                global_mesh = jax.sharding.Mesh(jax.devices(), "x")
                pspecs = jax.sharding.PartitionSpec("x")

                all_samples = host_local_array_to_global_array(σ1, global_mesh, pspecs)
                _log_prob = (
                    mpow
                    * apply_model(
                        machine, parameters, all_samples, sampler.chunk_size
                    ).real
                )
                assert len(_log_prob.addressable_shards) == 1
                _log_prob = global_array_to_host_local_array(
                    log_prob, global_mesh, pspecs
                )
            else:
                _log_prob = (
                    mpow * apply_model(machine, parameters, σ1, sampler.chunk_size).real
                )

            log_prob_1 = np.copy(_log_prob)
            assert log_prob_1.shape == σ1.shape[:-1]

            random_uniform = rgen.uniform(0, 1, size=σ.shape[0])

            # Acceptance Kernel
            accepted += acceptance_kernel(
                σ,
                σ1,
                log_prob,
                log_prob_1,
                log_prob_corr,
                mpow,
                random_uniform,
            )

        state.n_steps_proc += sampler.sweep_size * sampler.n_chains_per_rank
        state.n_accepted_proc += accepted

        return state, (state.σ, state.log_prob)

    def _sample_chain(
        sampler,
        machine: Callable,
        parameters: PyTree,
        state: MetropolisNumpySamplerState,
        chain_length: int,
        return_probabilties: bool = False,
    ) -> tuple[jnp.ndarray, MetropolisNumpySamplerState]:
        samples = np.empty(
            (chain_length, sampler.n_chains_per_rank, sampler.hilbert.size),
            dtype=sampler.dtype,
        )
        log_probs = np.empty((chain_length, sampler.n_chains_per_rank), dtype=float)

        for i in range(chain_length):
            state, (σ, log_prob) = sampler.sample_next(machine, parameters, state)
            samples[i] = σ
            log_probs[i] = log_prob

        # make it (n_chains_per_rank, n_samples_per_chain, hi.size) as expected by netket.stats.statistics
        samples = np.swapaxes(samples, 0, 1)
        log_probs = np.swapaxes(log_probs, 0, 1)

        if return_probabilties:
            return samples, log_probs, state
        else:
            return samples, state

    def __repr__(sampler):
        return (
            "MetropolisSamplerNumpy("
            + f"\n  hilbert = {sampler.hilbert},"
            + f"\n  rule = {sampler.rule},"
            + f"\n  n_chains = {sampler.n_chains},"
            + f"\n  machine_power = {sampler.machine_pow},"
            + f"\n  reset_chains = {sampler.reset_chains},"
            + f"\n  sweep_size = {sampler.sweep_size},"
            + f"\n  dtype = {sampler.dtype},"
            + ")"
        )

    def __str__(sampler):
        return (
            "MetropolisSamplerNumpy("
            + f"rule = {sampler.rule}, "
            + f"n_chains = {sampler.n_chains}, "
            + f"machine_power = {sampler.machine_pow}, "
            + f"sweep_size = {sampler.sweep_size}, "
            + f"dtype = {sampler.dtype})"
        )


@jit(nopython=True)
def acceptance_kernel(
    σ, σ1, log_prob, log_prob_1, log_prob_corr, machine_pow, random_uniform
):
    accepted = 0

    for i in range(σ.shape[0]):
        prob = np.exp(log_prob_1[i] - log_prob[i] + log_prob_corr[i])
        assert not math.isnan(prob)

        if prob > random_uniform[i]:
            log_prob[i] = log_prob_1[i]
            σ[i] = σ1[i]
            accepted += 1

    return accepted


def MetropolisLocalNumpy(hilbert: AbstractHilbert, **kwargs):
    from .rules import LocalRuleNumpy

    rule = LocalRuleNumpy()
    return MetropolisSamplerNumpy(hilbert, rule, **kwargs)


def MetropolisHamiltonianNumpy(hilbert: AbstractHilbert, hamiltonian, **kwargs):
    from .rules import HamiltonianRuleNumpy

    rule = HamiltonianRuleNumpy(hamiltonian)
    return MetropolisSamplerNumpy(hilbert, rule, **kwargs)


def MetropolisCustomNumpy(
    hilbert: AbstractHilbert, move_operators, move_weights=None, **kwargs
):
    from .rules import CustomRuleNumpy

    rule = CustomRuleNumpy(move_operators, move_weights)
    return MetropolisSamplerNumpy(hilbert, rule, **kwargs)
