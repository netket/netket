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

from typing import Any, Optional, Tuple, Callable
from functools import partial

import jax
from jax import numpy as jnp
from jax.experimental import host_callback as hcb

from netket.nn import to_array
from netket.utils import struct
from netket.utils.types import PyTree, SeedT

from .base import Sampler, SamplerState


@struct.dataclass
class ExactSamplerState(SamplerState):
    pdf: Any
    rng: Any

    def __repr__(self):
        return f"ExactSamplerState(rng state={self.rng})"


@struct.dataclass
class ExactSampler(Sampler):
    """
    This sampler generates i.i.d. samples from :math:`|\\Psi(\\sigma)|^2`.

    In order to perform exact sampling, :math:`|\\Psi(\\sigma)|^2` is precomputed an all
    the possible values of the quantum numbers :math:`\\sigma`. This sampler has thus an
    exponential cost with the number of degrees of freedom, and cannot be used
    for large systems, where Metropolis-based sampling are instead a viable
    option.
    """

    @property
    def is_exact(sampler):
        return True

    def _init_state(
        sampler,
        machine: Callable,
        parameters: PyTree,
        seed: Optional[SeedT] = None,
    ):
        pdf = jnp.zeros(sampler.hilbert.n_states, dtype=jnp.float32)
        return ExactSamplerState(pdf=pdf, rng=seed)

    def _reset(sampler, machine, parameters, state):
        pdf = jnp.absolute(
            to_array(sampler.hilbert, machine.apply, parameters) ** sampler.machine_pow
        )
        pdf = pdf / pdf.sum()

        return state.replace(pdf=pdf)

    def _sample_next(sampler, machine, parameters, state):
        new_rng, rng = jax.random.split(state.rng)
        numbers = jax.random.choice(
            rng,
            sampler.hilbert.n_states,
            shape=(sampler.n_chains_per_rank,),
            replace=True,
            p=state.pdf,
        )

        # We use a host-callback to convert integers labelling states to
        # valid state-arrays because that code is written with numba and
        # we have not yet converted it to jax.
        sample = hcb.call(
            lambda numbers: sampler.hilbert.numbers_to_states(numbers),
            numbers,
            result_shape=jax.ShapeDtypeStruct(
                (sampler.n_chains_per_rank, sampler.hilbert.size), jnp.float64
            ),
        )

        new_state = state.replace(rng=new_rng)
        return new_state, jnp.asarray(sample, dtype=sampler.dtype)

    def _sample_chain(
        sampler,
        machine: Callable,
        parameters: PyTree,
        state: SamplerState,
        chain_length: int,
    ) -> Tuple[jnp.ndarray, SamplerState]:
        # Reimplement sample_chain because we can sample the whole 'chain' in one
        # go, since it's not really a chain anyway. This will be much faster because
        # we call into python only once.
        return _sample_chain(sampler, machine, parameters, state, chain_length)

    def __repr__(sampler):
        return (
            "ExactSampler("
            + "\n  hilbert = {},".format(sampler.hilbert)
            + "\n  n_chains_per_rank = {},".format(sampler.n_chains_per_rank)
            + "\n  machine_power = {},".format(sampler.machine_pow)
            + "\n  dtype = {})".format(sampler.dtype)
        )

    def __str__(sampler):
        return (
            "ExactSampler("
            + "n_chains_per_rank = {}, ".format(sampler.n_chains_per_rank)
            + "machine_power = {}, ".format(sampler.machine_pow)
            + "dtype = {})".format(sampler.dtype)
        )


@partial(jax.jit, static_argnums=(1, 4))
def _sample_chain(
    sampler,
    machine: Callable,
    parameters: PyTree,
    state: SamplerState,
    chain_length: int,
) -> Tuple[jnp.ndarray, SamplerState]:
    new_rng, rng = jax.random.split(state.rng)
    numbers = jax.random.choice(
        rng,
        sampler.hilbert.n_states,
        shape=(chain_length * sampler.n_chains_per_rank,),
        replace=True,
        p=state.pdf,
    )

    # For future investigators:
    # this will lead to a crash if numbers_to_state throws.
    # it throws if we feed it nans!
    samples = hcb.call(
        lambda numbers: sampler.hilbert.numbers_to_states(numbers),
        numbers,
        result_shape=jax.ShapeDtypeStruct(
            (chain_length * sampler.n_chains_per_rank, sampler.hilbert.size),
            jnp.float64,
        ),
    )
    samples = jnp.asarray(samples, dtype=sampler.dtype).reshape(
        chain_length, sampler.n_chains_per_rank, sampler.hilbert.size
    )

    return samples, state.replace(rng=new_rng)
