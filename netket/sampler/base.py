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

import abc
from typing import Callable, Optional, Tuple, Union

import numpy as np
from flax import linen as nn
from jax import numpy as jnp

from netket import jax as nkjax
from netket.hilbert import AbstractHilbert
from netket.utils import mpi, get_afun_if_module, wrap_afun
from netket.utils.types import PyTree, DType, SeedT
from netket.jax import HashablePartial
from netket.utils import struct, numbers

fancy = []


@struct.dataclass
class SamplerState:
    """
    Base class holding the state of a sampler.
    """

    pass


def autodoc(clz):
    pass


@struct.dataclass
class Sampler(abc.ABC):
    """
    Abstract base class for all samplers.

    It contains the fields that all of them should possess, defining the common
    API.
    Note that fields marked with pytree_node=False are treated as static arguments
    when jitting.
    """

    hilbert: AbstractHilbert = struct.field(pytree_node=False)
    """The Hilbert space to sample."""
    _n_batches: Optional[int] = None
    """The batch size of the sampling on every MPI rank. Not implemented yet."""
    machine_pow: int = struct.field(default=2)
    """The power to which the machine should be exponentiated to generate the pdf."""
    dtype: DType = struct.field(pytree_node=False, default=np.float64)
    """The dtype of the states sampled."""

    def __pre_init__(self, *args, **kwargs):
        """
        Construct a Monte Carlo sampler.

        Args:
            hilbert: The Hilbert space to sample.
            n_batches: The batch size of the sampling on every MPI rank (default = None).
                Not implemented yet.
                If not specified, all the samples will be generated in a batch.
            machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
            dtype: The dtype of the states sampled (default = np.float64).
        """
        if "n_batches" in kwargs:
            kwargs["_n_batches"] = kwargs.pop("n_batches")
        return args, kwargs

    def __post_init__(self):
        # Raise errors if hilbert is not an Hilbert
        if not isinstance(self.hilbert, AbstractHilbert):
            raise ValueError(
                "hilbert must be a subtype of netket.hilbert.AbstractHilbert, "
                + "instead, type {} is not.".format(type(self.hilbert))
            )

        # workaround Jax bug under pmap
        # might be removed in the future
        if type(self.machine_pow) != object:
            if not np.issubdtype(numbers.dtype(self.machine_pow), np.integer):
                raise ValueError(
                    f"machine_pow ({self.machine_pow}) must be a positive integer"
                )

    @property
    def n_batches(self) -> Optional[int]:
        """
        The batch size of the sampling on every MPI rank. Not implemented yet.

        If not specified, all the samples will be generated in a batch.
        """
        return self._n_batches

    @property
    def is_exact(self) -> bool:
        """
        Returns `True` if the sampler is exact.

        The sampler is exact if all the samples are exactly distributed according to the
        chosen power of the variational state, and there is no correlation among them.
        """
        return False

    def log_pdf(self, model: Union[Callable, nn.Module]) -> Callable:
        """
        Returns a closure with the log_pdf function encoded by this sampler.

        Note: the result is returned as an HashablePartial so that the closure
        does not trigger recompilation.

        Args:
            model: The machine, or apply_fun

        Returns:
            the log probability density function
        """
        apply_fun = get_afun_if_module(model)
        log_pdf = HashablePartial(
            lambda apply_fun, pars, σ: self.machine_pow * apply_fun(pars, σ).real,
            apply_fun,
        )
        return log_pdf

    def init_state(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        seed: Optional[SeedT] = None,
    ) -> SamplerState:
        """
        Creates the structure holding the state of the sampler.

        If you want reproducible samples, you should specify `seed`, otherwise the state
        will be initialised randomly.

        If running across several MPI processes, all sampler_states are guaranteed to be
        in a different (but deterministic) state.
        This is achieved by first reducing (summing) the seed provided to every MPI rank,
        then generating n_rank seeds starting from the reduced one, and every rank is
        initialized with one of those seeds.

        The resulting state is guaranteed to be a frozen python dataclass (in particular,
        a flax's dataclass), and it can be serialized using Flax serialization methods.

        Args:
            machine: A Flax module or Callable with the forward pass of the log-pdf.
                If it's a Callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
            parameters: The PyTree of parameters of the model.
            seed: An optional seed or jax PRNGKey. If not specified, a random seed will be used.

        Returns:
            The structure holding the state of the sampler. In general you should not expect
            it to be in a valid state, and should reset it before use.
        """
        key = nkjax.PRNGKey(seed)
        key = nkjax.mpi_split(key)

        return sampler._init_state(wrap_afun(machine), parameters, key)

    def reset(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        state: Optional[SamplerState] = None,
    ) -> SamplerState:
        """
        Resets the state of the sampler. To be used every time the parameters are changed.

        Args:
            machine: A Flax module or Callable with the forward pass of the log-pdf.
                If it's a Callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler. If not specified, it will be constructed
                by calling :code:`sampler.init_state(machine, parameters)` with a random seed.

        Returns:
            A valid sampler state.
        """
        if state is None:
            state = sampler.init_state(machine, parameters)

        return sampler._reset(wrap_afun(machine), parameters, state)

    def sample(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        *,
        state: Optional[SamplerState] = None,
        n_samples: int = 1,
    ) -> Tuple[jnp.ndarray, SamplerState]:
        """
        Generate samples.

        Arguments:
            machine: A Flax module or Callable with the forward pass of the log-pdf.
                If it's a Callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler. If not specified, then initialize and reset it.
            n_samples: The total number of samples across all MPI ranks (default = 1).

        Returns:
            σ: The generated samples. Its shape depends on the type of the sampler.
                For exact samplers, the shape is `(n_samples, hilbert.size)`;
                For Markov chain samplers, the shape is `(n_chains, chain_length, hilbert.size)`.
            state: The new state of the sampler.
        """
        if state is None:
            state = sampler.reset(machine, parameters, state)

        n_samples_per_rank = max(int(np.ceil(n_samples / mpi.n_nodes)), 1)
        if mpi.n_nodes > 1 and mpi.rank == 0:
            if n_samples_per_rank * mpi.n_nodes != n_samples:
                import warnings

                warnings.warn(
                    f"Using {n_samples_per_rank} samples per rank among {mpi.n_nodes} ranks "
                    f"(total={n_samples_per_rank * mpi.n_nodes} instead of n_samples={n_samples}). "
                    "To silence this warning, use `n_samples` that is a multiple of the number of MPI ranks.",
                    category=UserWarning,
                )

        return sampler._sample(
            wrap_afun(machine), parameters, state, n_samples_per_rank
        )

    @abc.abstractmethod
    def _init_state(sampler, machine, parameters, seed) -> SamplerState:
        """
        Implementation of `init_state` for subclasses of `Sampler`.

        If you sub-class `Sampler`, you should override `_init_state` and not `init_state`
        itself, because `init_state` contains some common logic.
        """
        raise NotImplementedError("`_init_state` not implemented")

    @abc.abstractmethod
    def _reset(sampler, machine, parameters, state):
        """
        Implementation of `reset` for subclasses of `Sampler`.

        If you sub-class `Sampler`, you should override `_reset` and not `reset`
        itself, because `reset` contains some common logic.
        """
        raise NotImplementedError("`_reset` not implemented")

    @abc.abstractmethod
    def _sample(sampler, machine, parameters, state, n_samples_per_rank):
        """
        Implementation of `sample` for subclasses of `Sampler`.

        If you sub-class `Sampler`, you should override `_sample` and not `sample`
        itself, because `sample` contains some common logic.

        If using Jax, this function should be jitted.
        """
        raise NotImplementedError("`_sample` not implemented")


def sampler_state(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    seed: Optional[SeedT] = None,
) -> SamplerState:
    """
    Creates the structure holding the state of the sampler.

    If you want reproducible samples, you should specify `seed`, otherwise the state
    will be initialised randomly.

    If running across several MPI processes, all sampler_states are guaranteed to be
    in a different (but deterministic) state.
    This is achieved by first reducing (summing) the seed provided to every MPI rank,
    then generating n_rank seeds starting from the reduced one, and every rank is
    initialized with one of those seeds.

    The resulting state is guaranteed to be a frozen python dataclass (in particular,
    a flax's dataclass), and it can be serialized using Flax serialization methods.

    Args:
        sampler: The Monte Carlo sampler.
        machine: A Flax module or Callable with the forward pass of the log-pdf.
            If it's a Callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
        parameters: The PyTree of parameters of the model.
        seed: An optional seed or jax PRNGKey. If not specified, a random seed will be used.

    Returns:
        The structure holding the state of the sampler. In general you should not expect
        it to be in a valid state, and should reset it before use.
    """
    return sampler.init_state(machine, parameters, seed)


def reset(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    state: Optional[SamplerState] = None,
):
    """
    Resets the state of the sampler. To be used every time the parameters are changed.

    Args:
        sampler: The Monte Carlo sampler.
        machine: A Flax module or Callable with the forward pass of the log-pdf.
            If it's a Callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
        parameters: The PyTree of parameters of the model.
        state: The current state of the sampler. If not specified, it will be constructed
            by calling :code:`sampler.init_state(machine, parameters)` with a random seed.

    Returns:
        A valid sampler state.
    """
    sampler.reset(machine, parameters, state)


def sample(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    *,
    state: Optional[SamplerState] = None,
    n_samples: int = 1,
) -> Tuple[jnp.ndarray, SamplerState]:
    """
    Generate samples.

    Arguments:
        sampler: The Monte Carlo sampler.
        machine: A Flax module or Callable with the forward pass of the log-pdf.
            If it's a Callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
        parameters: The PyTree of parameters of the model.
        state: The current state of the sampler. If not specified, then initialize and reset it.
        n_samples: The total number of samples across all MPI ranks (default = 1).

    Returns:
        σ: The generated samples. Its shape depends on the type of the sampler.
            For exact samplers, the shape is `(n_samples, hilbert.size)`;
            For Markov chain samplers, the shape is `(n_chains, chain_length, hilbert.size)`.
        state: The new state of the sampler.
    """
    return sampler.sample(machine, parameters, state, n_samples)
