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
from typing import Optional, Union, Callable
from collections.abc import Iterator

import numpy as np
from flax import linen as nn
from jax import numpy as jnp

from netket import jax as nkjax
from netket.hilbert import AbstractHilbert
from netket.utils import get_afun_if_module, mpi, numbers, struct, wrap_afun
from netket.utils.deprecation import deprecated, warn_deprecation
from netket.utils.types import PyTree, DType, SeedT
from netket.jax import HashablePartial

fancy = []


@struct.dataclass
class SamplerState(abc.ABC):
    """
    Base class holding the state of a sampler.
    """


@struct.dataclass
class Sampler(abc.ABC):
    """
    Abstract base class for all samplers.

    It contains the fields that all of them should possess, defining the common
    API.
    Note that fields marked with `pytree_node=False` are treated as static arguments
    when jitting.

    Subclasses should be NetKet dataclasses and they should define the `_init_state`,
    `_reset` and `_sample_chain` methods which only accept positional arguments.
    See the respective method's definition for its signature.

    Notice that those methods are different from the API-entry point without the leading
    underscore in order to allow us to share some pre-processing code between samplers
    and simplify the definition of a new sampler.
    """

    hilbert: AbstractHilbert = struct.field(pytree_node=False)
    """The Hilbert space to sample."""

    n_chains: int = struct.field(pytree_node=False, default=None, repr=False)
    """Total Number of independent chains"""

    machine_pow: int = struct.field(default=2)
    """The power to which the machine should be exponentiated to generate the pdf."""

    dtype: DType = struct.field(pytree_node=False, default=float)
    """The dtype of the states sampled."""

    def __pre_init__(
        self, hilbert: AbstractHilbert, n_chains: Optional[int] = None, **kwargs
    ):
        """
        Construct a Monte Carlo sampler.

        Args:
            hilbert: The Hilbert space to sample.
            n_chains: The total number of independent chains across all MPI ranks (default = mpi.n_nodes)
            machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
            dtype: The dtype of the states sampled (default = np.float64).
        """

        if "n_chains_per_rank" in kwargs:
            warn_deprecation(
                "Specifying `n_chains_per_rank` when constructing the Sampler is deprecated."
            )
            if n_chains is None:
                n_chains = mpi.n_nodes * kwargs.pop("n_chains_per_rank")
            else:
                raise ValueError(
                    "Cannot specify both `n_chains` and `n_chains_per_rank`"
                )
        if n_chains is None:
            n_chains = mpi.n_nodes
        kwargs["n_chains"] = n_chains

        return (hilbert,), kwargs

    def __post_init__(self):
        # Raise error if n_chains is not a multiple of the number of mpi ranks
        self.n_chains_per_rank

        # Raise errors if hilbert is not an Hilbert
        if not isinstance(self.hilbert, AbstractHilbert):
            raise TypeError(
                "\n\nThe argument `hilbert` of a Sampler must be a subtype "
                "of netket.hilbert.AbstractHilbert, but you passed in an object "
                f"of type {type(self.hilbert)}, which is not an AbstractHilbert.\n\n"
                "TO FIX THIS ERROR,\ndouble check the arguments passed to the "
                "sampler when constructing it, and verify that they have the "
                "correct types.\n\n"
                "For more information, check the correct arguments in the API "
                "reference at https://netket.readthedocs.io/en/latest/api/sampler.html"
                "\n"
            )

        # workaround Jax bug under pmap
        # might be removed in the future
        if type(self.machine_pow) != object:
            if not np.issubdtype(numbers.dtype(self.machine_pow), np.integer):
                raise ValueError(
                    f"machine_pow ({self.machine_pow}) must be a positive integer"
                )

    @property
    def n_chains_per_rank(self) -> int:
        """
        The total number of independent chains per MPI rank.

        If you are not using MPI, this is equal to :attr:`~Sampler.n_chains`.
        """
        res, remainder = divmod(self.n_chains, mpi.n_nodes)
        if remainder != 0:
            raise RuntimeError(
                "The number of chains is not a multiple of the number of mpi ranks"
            )
        return res

    @property
    def n_batches(self) -> int:
        r"""
        The batch size of the configuration $\sigma$ used by this sampler.

        In general, it is equivalent to :attr:`~Sampler.n_chains_per_rank`.
        """
        return self.n_chains_per_rank

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
        Returns a closure with the log-pdf function encoded by this sampler.

        Args:
            model: A Flax module or callable with the forward pass of the log-pdf.
                If it is a callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.

        Returns:
            The log-probability density function.

        Note:
            The result is returned as a `HashablePartial` so that the closure
            does not trigger recompilation.
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

        If running across several MPI processes, all `sampler_state`s are guaranteed to be
        in a different (but deterministic) state.
        This is achieved by first reducing (summing) the seed provided to every MPI rank,
        then generating `n_rank` seeds starting from the reduced one, and every rank is
        initialized with one of those seeds.

        The resulting state is guaranteed to be a frozen Python dataclass (in particular,
        a Flax dataclass), and it can be serialized using Flax serialization methods.

        Args:
            machine: A Flax module or callable with the forward pass of the log-pdf.
                If it is a callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
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
            machine: A Flax module or callable with the forward pass of the log-pdf.
                If it is a callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
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
        chain_length: int = 1,
    ) -> tuple[jnp.ndarray, SamplerState]:
        """
        Samples `chain_length` batches of samples along the chains.

        Arguments:
            machine: A Flax module or callable with the forward pass of the log-pdf.
                If it is a callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler. If not specified, then initialize and reset it.
            chain_length: The length of the chains (default = 1).

        Returns:
            σ: The generated batches of samples.
            state: The new state of the sampler.
        """
        if state is None:
            state = sampler.reset(machine, parameters)

        return sampler._sample_chain(
            wrap_afun(machine), parameters, state, chain_length
        )

    def samples(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        *,
        state: Optional[SamplerState] = None,
        chain_length: int = 1,
    ) -> Iterator[jnp.ndarray]:
        """
        Returns a generator sampling `chain_length` batches of samples along the chains.

        Arguments:
            machine: A Flax module or callable with the forward pass of the log-pdf.
                If it is a callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler. If not specified, then initialize and reset it.
            chain_length: The length of the chains (default = 1).
        """
        if state is None:
            state = sampler.reset(machine, parameters)

        machine = wrap_afun(machine)

        for _i in range(chain_length):
            samples, state = sampler._sample_chain(machine, parameters, state, 1)
            yield samples[:, 0, :]

    @abc.abstractmethod
    def _sample_chain(
        sampler,
        machine: nn.Module,
        parameters: PyTree,
        state: SamplerState,
        chain_length: int,
    ) -> tuple[jnp.ndarray, SamplerState]:
        """
        Implementation of `sample` for subclasses of `Sampler`.

        If you subclass `Sampler`, you should override this and not `sample`
        itself, because `sample` contains some common logic.

        If using Jax, this function should be jitted.

        Arguments:
            machine: A Flax module with the forward pass of the log-pdf.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler.
            chain_length: The length of the chains.

        Returns:
            σ: The generated batches of samples.
            state: The new state of the sampler.
        """

    @abc.abstractmethod
    def _init_state(sampler, machine, params, seed) -> SamplerState:
        """
        Implementation of `init_state` for subclasses of `Sampler`.

        If you subclass `Sampler`, you should override this and not `init_state`
        itself, because `init_state` contains some common logic.
        """

    @abc.abstractmethod
    def _reset(sampler, machine, parameters, state):
        """
        Implementation of `reset` for subclasses of `Sampler`.

        If you subclass `Sampler`, you should override this and not `reset`
        itself, because `reset` contains some common logic.
        """


@deprecated(
    "The module function `sampler_state` is deprecated in favor of the class method `init_state`."
)
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

    If running across several MPI processes, all `sampler_state`s are guaranteed to be
    in a different (but deterministic) state.
    This is achieved by first reducing (summing) the seed provided to every MPI rank,
    then generating `n_rank` seeds starting from the reduced one, and every rank is
    initialized with one of those seeds.

    The resulting state is guaranteed to be a frozen Python dataclass (in particular,
    a Flax dataclass), and it can be serialized using Flax serialization methods.

    Args:
        sampler: The Monte Carlo sampler.
        machine: A Flax module or callable with the forward pass of the log-pdf.
            If it is a callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
        parameters: The PyTree of parameters of the model.
        seed: An optional seed or jax PRNGKey. If not specified, a random seed will be used.

    Returns:
        The structure holding the state of the sampler. In general you should not expect
        it to be in a valid state, and should reset it before use.
    """
    return sampler.init_state(machine, parameters, seed)


@deprecated(
    "The module function `reset` is deprecated in favor of the class method `reset`."
)
def reset(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    state: Optional[SamplerState] = None,
) -> SamplerState:
    """
    Resets the state of the sampler. To be used every time the parameters are changed.

    Args:
        sampler: The Monte Carlo sampler.
        machine: A Flax module or callable with the forward pass of the log-pdf.
            If it is a callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
        parameters: The PyTree of parameters of the model.
        state: The current state of the sampler. If not specified, it will be constructed
            by calling :code:`sampler.init_state(machine, parameters)` with a random seed.

    Returns:
        A valid sampler state.
    """
    return sampler.reset(machine, parameters, state)


@deprecated(
    "The module function `sample` is deprecated in favor of the class method `sample`."
)
def sample(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    *,
    state: Optional[SamplerState] = None,
    chain_length: int = 1,
) -> tuple[jnp.ndarray, SamplerState]:
    """
    Samples `chain_length` batches of samples along the chains.

    Arguments:
        sampler: The Monte Carlo sampler.
        machine: A Flax module or callable with the forward pass of the log-pdf.
            If it is a callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
        parameters: The PyTree of parameters of the model.
        state: The current state of the sampler. If not specified, then initialize and reset it.
        chain_length: The length of the chains (default = 1).

    Returns:
        σ: The generated batches of samples.
        state: The new state of the sampler.
    """
    return sampler.sample(machine, parameters, state=state, chain_length=chain_length)


@deprecated(
    "The module function `samples` is deprecated in favor of the class method `samples`."
)
def samples(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    *,
    state: Optional[SamplerState] = None,
    chain_length: int = 1,
) -> Iterator[jnp.ndarray]:
    """
    Returns a generator sampling `chain_length` batches of samples along the chains.

    Arguments:
        sampler: The Monte Carlo sampler.
        machine: A Flax module or callable with the forward pass of the log-pdf.
            If it is a callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
        parameters: The PyTree of parameters of the model.
        state: The current state of the sampler. If not specified, then initialize and reset it.
        chain_length: The length of the chains (default = 1).
    """
    yield from sampler.samples(
        machine, parameters, state=state, chain_length=chain_length
    )
