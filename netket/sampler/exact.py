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

import jax
from jax import numpy as jnp
from jax.experimental import host_callback as hcb

from flax import struct

from typing import Any

from netket.nn import to_array
from netket.hilbert import AbstractHilbert

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

    def _init_state(sampler, machine, params, key):
        pdf = jnp.zeros(sampler.hilbert.n_states, dtype=jnp.float32)
        return ExactSamplerState(pdf=pdf, rng=key)

    def _reset(sampler, machine, parameters, state):
        pdf = jnp.absolute(
            to_array(sampler.hilbert, machine, parameters) ** sampler.machine_pow
        )
        pdf = pdf / pdf.sum()

        return state.replace(pdf=pdf)

    def _sample_next(sampler, machine, parameters, state):
        new_rng, rng = jax.random.split(state.rng)
        numbers = jax.random.choice(
            rng,
            sampler.hilbert.n_states,
            shape=(sampler.n_chains,),
            replace=True,
            p=state.pdf,
        )

        # We use a host-callback to convert integers labelling states to
        # valid state-arrays because that code is written with numba and
        # we have not yet converted it to jax.
        cb = lambda numbers: host_numbers_to_states(sampler.hilbert, numbers)

        sample = hcb.call(
            cb,
            numbers,
            result_shape=jax.ShapeDtypeStruct(
                (sampler.n_chains, sampler.hilbert.size), jnp.float64
            ),
        )

        new_state = state.replace(rng=new_rng)
        return new_state, jnp.asarray(sample, dtype=sampler.dtype)

    def __repr__(sampler):
        return (
            "ExactSampler("
            + "\n  hilbert = {},".format(sampler.hilbert)
            + "\n  n_chains = {},".format(sampler.n_chains)
            + "\n  machine_power = {},".format(sampler.machine_pow)
            + "\n  dtype = {})".format(sampler.dtype)
        )

    def __str__(sampler):
        return (
            "ExactSampler("
            + "n_chains = {}, ".format(sampler.n_chains)
            + "machine_power = {}, ".format(sampler.machine_pow)
            + "dtype = {})".format(sampler.dtype)
        )


from netket.legacy.sampler import ExactSampler as LegacyExactSampler
from netket.legacy.machine import AbstractMachine
from netket.utils import wraps_legacy


def host_numbers_to_states(hilbert, numbers):
    return hilbert.numbers_to_states(numbers)
