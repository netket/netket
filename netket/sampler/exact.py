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

from typing import Any, Optional
from functools import partial

import jax
from flax import linen as nn
from jax import numpy as jnp

from netket import config
from netket.hilbert import DiscreteHilbert
from netket.nn import to_array
from netket.utils.types import PyTree, SeedT, DType

from .base import Sampler, SamplerState


class ExactSamplerState(SamplerState):
    pdf: Any
    rng: Any

    def __init__(self, pdf: Any, rng: Any):
        self.pdf = pdf
        self.rng = rng
        super().__init__()

    def __repr__(self):
        return f"ExactSamplerState(rng state={self.rng})"


class ExactSampler(Sampler):
    """
    This sampler generates i.i.d. samples from :math:`|\\Psi(\\sigma)|^2`.

    In order to perform exact sampling, :math:`|\\Psi(\\sigma)|^2` is precomputed an all
    the possible values of the quantum numbers :math:`\\sigma`. This sampler has thus an
    exponential cost with the number of degrees of freedom, and cannot be used
    for large systems, where Metropolis-based sampling are instead a viable
    option.
    """

    def __init__(
        self,
        hilbert: DiscreteHilbert,
        machine_pow: int = 2,
        dtype: DType = float,
    ):
        """
        Construct an exact sampler.

        Args:
            hilbert: The Hilbert space to sample.
            machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
            dtype: The dtype of the states sampled (default = np.float64).
        """
        super().__init__(hilbert, machine_pow=machine_pow, dtype=dtype)

    @property
    def is_exact(sampler):
        return True

    def _init_state(
        sampler,
        machine: nn.Module,
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

    @partial(jax.jit, static_argnums=(1, 4))
    def _sample_chain(
        sampler,
        machine: nn.Module,
        parameters: PyTree,
        state: SamplerState,
        chain_length: int,
    ) -> tuple[jnp.ndarray, SamplerState]:
        # Reimplement sample_chain because we can sample the whole 'chain' in one
        # go, since it's not really a chain anyway. This will be much faster because
        # we call into python only once.
        new_rng, rng = jax.random.split(state.rng)
        numbers = jax.random.choice(
            rng,
            sampler.hilbert.n_states,
            shape=(chain_length * sampler.n_chains_per_rank,),
            replace=True,
            p=state.pdf,
        )

        # We use a host-callback to convert integers labelling states to
        # valid state-arrays because that code is written with numba and
        # we have not yet converted it to jax.
        #
        # For future investigators:
        # this will lead to a crash if numbers_to_state throws.
        # it throws if we feed it nans!
        samples = jax.pure_callback(
            lambda numbers: sampler.hilbert.numbers_to_states(numbers).astype(
                sampler.dtype
            ),
            jax.ShapeDtypeStruct(
                (sampler.n_chains_per_rank * chain_length, sampler.hilbert.size),
                sampler.dtype,
            ),
            numbers,
        )
        samples = jnp.asarray(samples, dtype=sampler.dtype).reshape(
            sampler.n_chains_per_rank, chain_length, sampler.hilbert.size
        )

        # TODO run the part above in parallel
        if config.netket_experimental_sharding:
            samples = jax.lax.with_sharding_constraint(
                samples,
                jax.sharding.PositionalSharding(jax.devices()).reshape(1, -1, 1),
            )

        return samples, state.replace(rng=new_rng)
