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

from typing import Callable, Union, Tuple
from functools import partial

import jax
from jax import numpy as jnp

import numpy as np

from flax import linen as nn

from netket.utils import mpi, config, struct
from netket.utils.types import PyTree

from .base import SamplerState
from .metropolis import MetropolisSamplerState, MetropolisSampler


@struct.dataclass
class MetropolisSamplerPmapState(MetropolisSamplerState):
    """
    Sampler State for the `MetropolisSamplerPmap` sampler.

    Wraps `MetropolisSamplerState` and overrides a few properties to work
    with the ShardedDeviceArrays.
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
    r"""
    Metropolis-Hastings sampler for an Hilbert space according to a specific transition rule where chains are split
    among the available devices (`jax.devices()`).

    This sampler is experimental. It's API might change without warnings in future NetKet releases.

    To parallelize on CPU, you should set the following environment variable before loading jax/NetKet,
    XLA_FLAGS="--xla_force_host_platform_device_count=XX", where XX is the number of desired cpu devices.

    The transition rule is used to generate a proposed state :math:`s^\prime`, starting from the
    current state :math:`s`. The move is accepted with probability

    .. math::

        A(s \rightarrow s^\prime) = \mathrm{min} \left( 1,\frac{P(s^\prime)}{P(s)} F(e^{L(s,s^\prime)}) \right) ,

    where the probability being sampled from is :math:`P(s)=|M(s)|^p. Here ::math::`M(s)` is a
    user-provided function (the machine), :math:`p` is also user-provided with default value :math:`p=2`,
    and :math:`L(s,s^\prime)` is a suitable correcting factor computed by the transition kernel.

    The dtype of the sampled states can be chosen.
    """

    def __pre_init__(self, *args, n_chains_per_device=None, **kwargs):
        """
        Constructs a Metropolis Sampler.

        Args:
            hilbert: The hilbert space to sample
            rule: A `MetropolisRule` to generate random transitions from a given state as
                    well as uniform random states.
            n_sweeps: The number of exchanges that compose a single sweep.
                    If None, sweep_size is equal to the number of degrees of freedom being sampled
                    (the size of the input vector s to the machine).
            reset_chains: If False the state configuration is not resetted when reset() is called.
            n_chains: The total number of Markov Chain to be run in parallel on a the available devices.
                This will be rounded to the nearest multiple of `len(jax.devices())`
            n_chains_per_device: The number of chains to be run in parallel on one device.
                Cannot be specified if n_chains is also specified.
            machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
            dtype: The dtype of the statees sampled (default = np.float32).
        """

        n_devices = len(jax.devices())

        n_chains_undef = "n_chains" not in kwargs and "n_chains_per_rank" not in kwargs

        if not n_chains_undef and n_chains_per_device is not None:
            raise ValueError(
                "Cannot specify both n_chains/n_chains_per_rank and n_chains_per_device"
            )

        args, kwargs = super().__pre_init__(*args, **kwargs)

        if n_chains_per_device is None:
            n_chains_per_device = int(
                max(np.ceil(kwargs["n_chains_per_rank"] / n_devices), 1)
            )

        kwargs["n_chains_per_rank"] = n_chains_per_device * n_devices

        if kwargs["n_chains_per_rank"] != n_chains_per_device * n_devices:
            import warnings

            warnings.warn(
                f"Using {n_chains_per_device*n_devices} chains "
                f"({n_chains_per_device} chains on each of {n_devices} devices).",
                category=UserWarning,
            )

        return args, kwargs

    def __post_init__(self):
        if not config.FLAGS["NETKET_EXPERIMENTAL"]:
            raise RuntimeError(
                """
                               The Pmapped Metropolis sampler is an experimental
                               feature. We have not yet extensively investigated how it affects
                               performance, and when it is appropriate to use it.

                               The API is experimental, and might change without warnings in
                               future NetKet releases.

                               Use it at your own risk by setting the environment variable
                               NETKET_EXPERIMENTAL=1
                               """
            )

        super().__post_init__()

        n_chains_per_device = self.n_chains_per_rank // len(jax.devices())

        _sampler = MetropolisSampler(
            self.hilbert,
            n_chains_per_rank=n_chains_per_device,
            rule=self.rule,
            n_sweeps=self.n_sweeps,
            reset_chains=self.reset_chains,
            machine_pow=self.machine_pow,
        )

        object.__setattr__(self, "_sampler_device", _sampler)

    @property
    def n_chains_per_device(self):
        return self._sampler_device.n_chains_per_rank

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
