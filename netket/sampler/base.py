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
from typing import Optional, Union, Tuple, Callable, Iterator
from functools import partial

import jax
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

    It contains the fields that all of them should posses, defining the common
    API.
    Note that fields marked with pytree_node=False are treated as static arguments
    when jitting.
    """

    hilbert: AbstractHilbert = struct.field(pytree_node=False)
    """Hilbert space to be sampled."""

    n_chains_per_rank: int = struct.field(pytree_node=False, default=None)
    """Number of independent chains on every MPI rank."""

    machine_pow: int = struct.field(default=2)
    """Exponent of the pdf sampled."""

    dtype: DType = struct.field(pytree_node=False, default=np.float64)
    """DType of the computed samples."""

    def __pre_init__(
        self, hilbert: AbstractHilbert, n_chains: Optional[int] = None, **kwargs
    ):
        """
        Construct a Monte Carlo sampler.

        Args:
            hilbert: Hilbert space to be sampled
            n_chains: The total number of independent chains across all MPI ranks
            n_chains_per_rank: Number of independent chains on every MPI rank
            machine_pow: Exponent of the pdf sampled
            dtype: DType of the computed samples
        """

        # Default number of total chains
        if "n_chains_per_rank" in kwargs:
            if n_chains is not None:
                raise ValueError(
                    "Cannot specify both `n_chains` and `n_chains_per_rank`"
                )
        else:

            # DEFAULT VALUE
            if n_chains is None:
                n_chains = 16

            n_chains_per_rank = max(int(np.ceil(n_chains / mpi.n_nodes)), 1)
            if mpi.n_nodes > 1 and mpi.rank == 0:
                if n_chains_per_rank * mpi.n_nodes != n_chains:
                    import warnings

                    warnings.warn(
                        f"Using {n_chains_per_rank} chains per rank among {mpi.n_nodes} ranks (total="
                        f"{n_chains_per_rank*mpi.n_nodes} instead of n_chains={n_chains})."
                        f"To directly control the number of chains on every rank, specify "
                        f"`n_chains_per_rank` when constructing the sampler. "
                        f"To silence this warning, either use `n_chains_per_rank` or use `n_chains` "
                        f"that is a multiple of the number of mpi ranks.",
                        category=UserWarning,
                    )

            kwargs["n_chains_per_rank"] = n_chains_per_rank

        return (hilbert,), kwargs

    def __post_init__(self):
        # Raise errors if hilbert is not an Hilbert
        if not isinstance(self.hilbert, AbstractHilbert):
            raise ValueError(
                "hilbert must be a subtype of netket.hilbert.AbstractHilbert, "
                + "instead, type {} is not.".format(type(self.hilbert))
            )

        # workaround Jax bug under pmap
        # might be removed in the future
        if not type(self.machine_pow) == object:
            if not np.issubdtype(numbers.dtype(self.machine_pow), np.integer):
                raise ValueError(
                    f"machine_pow ({self.machine_pow}) must be a positive integer"
                )

    @property
    def n_chains(self) -> int:
        """
        The total number of chains across all MPI ranks.

        If you are not using MPI, this is equal to `n_chains_per_rank`
        """
        return self.n_chains_per_rank * mpi.n_nodes

    @property
    def n_batches(self) -> int:
        r"""
        The batch size of the configuration $\sigma$ used by this sampler.

        In general, it is equivalent to :attr:`~Sampler.n_chains`.
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
            machine: a Flax module or callable with the forward pass of the log-pdf.
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
            machine: a Flax module or callable with the forward pass of the log-pdf.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler. If it's not provided, it will be constructed
                by calling :code:`sampler.init_state(machine, parameters)` with a random seed.

        Returns:
            A valid sampler state.
        """
        if state is None:
            state = sampler_state(sampler, machine, parameters)

        return sampler._reset(wrap_afun(machine), parameters, state)

    def sample_next(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        state: Optional[SamplerState] = None,
    ) -> Tuple[jnp.ndarray, SamplerState]:
        """
        Samples the next state in the markov chain.

        Args:
            machine: a Flax module or callable apply function with the forward pass of the log-pdf.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler. If it's not provided, it will be constructed
                by calling :code:`sampler.reset(machine, parameters)` with a random seed.

        Returns:
            state: The new state of the sampler
            σ: The next batch of samples.
        """
        # Note: the return order is inverted wrt `.sample` because when called inside of
        # a scan function the first returned argument should be the state.

        if state is None:
            state = sampler_state(sampler, machine, parameters)

        return sampler._sample_next(wrap_afun(machine), parameters, state)

    def sample(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        *,
        state: Optional[SamplerState] = None,
        chain_length: int = 1,
    ) -> Tuple[jnp.ndarray, SamplerState]:
        """
        Samples chain_length elements along the chains.

        Arguments:
            sampler: The Monte Carlo sampler.
            machine: The model or callable to sample from (if it's a function it should have
                the signature :code:`f(parameters, σ) -> jnp.ndarray`).
            parameters: The PyTree of parameters of the model.
            state: current state of the sampler. If None, then initialises it.
            chain_length: (default=1), the length of the chains.

        Returns:
            σ: The next batch of samples.
            state: The new state of the sampler
        """

        return sample(
            sampler, machine, parameters, state=state, chain_length=chain_length
        )

    def _sample_chain(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        state: SamplerState,
        chain_length: int,
    ) -> Tuple[jnp.ndarray, SamplerState]:
        """
        Samples chain_length elements along the chains.

        In general this should not be overridden unless you want to modify the logic by which
        the whole sampling is performed.
        If using Jax, this function should be jitted

        Arguments:
            sampler: The Monte Carlo sampler.
            machine: The model or callable to sample from (if it's a function it should have
                the signature :code:`f(parameters, σ) -> jnp.ndarray`).
            parameters: The PyTree of parameters of the model.
            state: current state of the sampler. If None, then initialises it.
            chain_length: (default=1), the length of the chains.

        Returns:
            σ: The next batch of samples.
            state: The new state of the sampler
        """
        return _sample_chain(sampler, machine, parameters, state, chain_length)

    @abc.abstractmethod
    def _init_state(sampler, machine, params, seed) -> SamplerState:
        """
        Implementation of init_state for subclasses of Sampler.

        If you sub-class Sampler, you should define this and not init_state
        itself, because init_state contains some common logic.
        """
        raise NotImplementedError("init_state Not Implemented")

    @abc.abstractmethod
    def _reset(sampler, machine, parameters, state):
        """
        Implementation of reset for subclasses of Sampler.

        If you sub-class Sampler, you should define _reset and not reset
        itself, because reset contains some common logic.
        """
        raise NotImplementedError("reset Not Implemented")

    @abc.abstractmethod
    def _sample_next(sampler, machine, parameters, state=None):
        """
        Implementation of sample_next for subclasses of Sampler.

        If you sub-class Sampler, you should define _sample_next and not sample_next
        itself, because reset contains some common logic.
        """
        raise NotImplementedError("sample_next Not Implemented")


def sampler_state(
    sampler: Sampler, machine: Union[Callable, nn.Module], parameters: PyTree
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

    Args:
        sampler: The Monte Carlo sampler.
        machine: a Flax module or callable with the forward pass of the log-pdf.
        parameters: The PyTree of parameters of the model.
        seed: An optional seed or jax PRNGKey. If not specified, a random seed will be used.

    Returns:
        The structure holding the state of the sampler. In general you should not expect
        it to be in a valid state, and should reset it before use.
    """
    return sampler.init_state(machine, parameters)


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
        machine: a Flax module or Callable with the forward pass of the log-pdf.
        parameters: The PyTree of parameters of the model.
        state: The current state of the sampler. If it's not provided, it will be constructed
            by calling :code:`sampler.init_state(machine, parameters)` with a random seed.

    Returns:
        A valid sampler state.
    """
    sampler.reset(machine, parameters, state)


def sample_next(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    state: Optional[SamplerState] = None,
) -> Tuple[jnp.ndarray, SamplerState]:
    """
    Samples the next state in the markov chain.

    Args:
        sampler: The Monte Carlo sampler.
        machine: a Flax module or callable with the forward pass of the log-pdf.
        parameters: The PyTree of parameters of the model.
        state: The current state of the sampler. If it's not provided, it will be constructed
            by calling :code:`sampler.reset(machine, parameters)` with a random seed.

    Returns:
        state: The new state of the sampler
        σ: The next batch of samples.
    """
    return sampler.sample_next(machine, parameters, state)


def sample(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    *,
    state: Optional[SamplerState] = None,
    chain_length: int = 1,
) -> Tuple[jnp.ndarray, SamplerState]:
    """
    Samples chain_length elements along the chains.

    Arguments:
        sampler: The Monte Carlo sampler.
        machine: The model or Callable to sample from (if it's a function it should have
            the signature :code:`f(parameters, σ) -> jnp.ndarray`).
        parameters: The PyTree of parameters of the model.
        state: current state of the sampler. If None, then initialises it.
        chain_length: (default=1), the length of the chains.

    Returns:
        σ: The next batch of samples.
        state: The new state of the sampler
    """
    if state is None:
        state = sampler.reset(machine, parameters, state)

    return sampler._sample_chain(machine, parameters, state, chain_length)


@partial(jax.jit, static_argnums=(1, 4))
def _sample_chain(
    sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    state: SamplerState,
    chain_length: int,
) -> Tuple[jnp.ndarray, SamplerState]:
    """
    Samples chain_length elements along the chains.

    Internal method used for jitting calls

    Arguments:
        sampler: The Monte Carlo sampler.
        machine: The model or Callable to sample from (if it's a function it should have
            the signature :code:`f(parameters, σ) -> jnp.ndarray`).
        parameters: The PyTree of parameters of the model.
        state: current state of the sampler. If None, then initialises it.
        chain_length: (default=1), the length of the chains.

    Returns:
        σ: The next batch of samples.
        state: The new state of the sampler
    """
    _sample_next = lambda state, _: sampler.sample_next(machine, parameters, state)

    state, samples = jax.lax.scan(
        _sample_next,
        state,
        xs=None,
        length=chain_length,
    )

    return samples, state


def samples(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    *,
    state: Optional[SamplerState] = None,
    chain_length: int = 1,
) -> Iterator[np.ndarray]:
    """
    Returns a generator sampling chain_length elements along the chains.

    Arguments:
        sampler: The Monte Carlo sampler.
        machine: The model or Callable to sample from (if it's a function it should have
            the signature :code:`f(parameters, σ) -> jnp.ndarray`).
        parameters: The PyTree of parameters of the model.
        state: current state of the sampler. If None, then initialises it.
        chain_length: (default=1), the length of the chains.
    """
    if state is None:
        state = sampler.reset(machine, parameters, state)

    for i in range(chain_length):
        samples, state = sampler._sample_chain(machine, parameters, state, 1)
        yield samples[0, :, :]
