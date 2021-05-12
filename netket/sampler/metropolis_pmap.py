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

from typing import Any, Optional, Callable, Union, Tuple
from functools import partial

import jax
from jax import numpy as jnp
from jax.experimental import loops

import numpy as np

from flax import struct
from flax import linen as nn

from netket.hilbert import AbstractHilbert
from netket.utils import mpi
from netket.utils.types import PyTree, PRNGKeyT

from netket.utils.deprecation import deprecated, warn_deprecation

from .base import Sampler, SamplerState
from .metropolis import MetropolisRule, MetropolisSamplerState, MetropolisSampler


@struct.dataclass
class MetropolisSamplerPmapState(MetropolisSamplerState):
    """
    Sampler State for the `MetropolisSamplerPmap` sampler.
    Wraps `MetropolisSamplerState` and overrides a few properties to work
    with the ShardedDeviceArrays
    """

    @property
    def n_steps(self) -> int:
        return self.n_steps_proc.sum() * mpi.n_nodes

    @property
    def n_accepted(self) -> int:
        """Total number of moves accepted across all processes since the last reset."""
        res, _ = mpi.mpi_sum_jax(self.n_accepted_proc.sum())
        return res

    def _repr_pretty_(self, p, cycle):
        super()._repr_pretty_(p, cycle)


@struct.dataclass
class MetropolisSamplerPmap(MetropolisSampler):
    """
    Metropolis-Hastings sampler for an Hilbert space according to a specific transition rule where chains are split
    among the available devices (`jax.devices()`).

    To parallelize on CPU, you should set the following environment variable before loading jax/NetKet,
    XLA_FLAGS="--xla_force_host_platform_device_count=XX", where XX is the number of desired cpu devices.
    .

    The transition rule is used to generate a proposed state :math:`s^\prime`, starting from the
    current state :math:`s`. The move is accepted with probability

    .. math::

        A(s \\rightarrow s^\\prime) = \\mathrm{min} \\left( 1,\\frac{P(s^\\prime)}{P(s)} F(e^{L(s,s^\\prime)}) \\right) ,

    where the probability being sampled from is :math:`P(s)=|M(s)|^p. Here ::math::`M(s)` is a
    user-provided function (the machine), :math:`p` is also user-provided with default value :math:`p=2`,
    and :math:`L(s,s^\prime)` is a suitable correcting factor computed by the transition kernel.

    The dtype of the sampled states can be chosen.
    """

    def __post_init__(self):
        super().__post_init__()

        n_devices = len(jax.devices())

        n_chains_per_device = int(max(np.ceil(self.n_chains / n_devices), 1))

        _sampler = MetropolisSampler(
            self.hilbert,
            n_chains=n_chains_per_device,
            rule=self.rule,
            n_sweeps=self.n_sweeps,
            reset_chains=self.reset_chains,
        )

        object.__setattr__(self, "n_chains", n_chains_per_device * n_devices)
        object.__setattr__(self, "_sampler_device", _sampler)

    def _init_state(self, machine, parameters, key):
        key = jax.random.split(key, len(jax.devices()))
        state = _init_state_pmap(self._sampler_device, machine, parameters, key)
        state = MetropolisSamplerPmapState(
            σ=state.σ,
            rng=state.rng,
            rule_state=state.rule_state,
            n_steps_proc=state.n_steps_proc,
            n_accepted_proc=state.n_accepted_proc,
        )
        return state

    def _reset(self, machine, parameters, state):
        state = _reset_pmap(self._sampler_device, machine, parameters, state)
        state = MetropolisSamplerPmapState(
            σ=state.σ,
            rng=state.rng,
            rule_state=state.rule_state,
            n_steps_proc=state.n_steps_proc,
            n_accepted_proc=state.n_accepted_proc,
        )
        return state

    def _sample_next(self, machine, parameters, state):
        state, conf = _sample_next_pmap(
            self._sampler_device, machine, parameters, state
        )
        state = MetropolisSamplerPmapState(
            σ=state.σ,
            rng=state.rng,
            rule_state=state.rule_state,
            n_steps_proc=state.n_steps_proc,
            n_accepted_proc=state.n_accepted_proc,
        )
        return state, conf.reshape(-1, conf.shape[-1])

    def _sample_chain(
        self,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        state: SamplerState,
        chain_length: int,
    ) -> Tuple[jnp.ndarray, SamplerState]:
        samples, state = _sample_chain_pmap(
            self._sampler_device, machine, parameters, state, chain_length
        )
        state = MetropolisSamplerPmapState(
            σ=state.σ,
            rng=state.rng,
            rule_state=state.rule_state,
            n_steps_proc=state.n_steps_proc,
            n_accepted_proc=state.n_accepted_proc,
        )
        return samples.reshape(samples.shape[0], -1, samples.shape[-1]), state

    def _repr_pretty_(sampler, p, cycle):
        super()._repr_pretty_(p, cycle)


@partial(
    jax.pmap, in_axes=(None, None, None, 0), out_axes=0, static_broadcasted_argnums=1
)
def _init_state_pmap(sampler, machine, parameters, key):
    return sampler._init_state(machine, parameters, key)


@partial(
    jax.pmap, in_axes=(None, None, None, 0), out_axes=0, static_broadcasted_argnums=1
)
def _reset_pmap(sampler, machine, parameters, state):
    return sampler._reset(machine, parameters, state)


@partial(
    jax.pmap,
    in_axes=(None, None, None, 0),
    out_axes=(0, 0),
    static_broadcasted_argnums=1,
)
def _sample_next_pmap(sampler, machine, parameters, state):
    return sampler._sample_next(machine, parameters, state)


@partial(
    jax.pmap,
    in_axes=(None, None, None, 0, None),
    out_axes=(1, 0),
    static_broadcasted_argnums=(1, 4),
)
def _sample_chain_pmap(sampler, machine, parameters, state, chain_length):
    return sampler._sample_chain(machine, parameters, state, chain_length)
