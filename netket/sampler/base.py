import abc
from typing import Optional, Any, Union, Tuple, Callable
from functools import singledispatch, partial

import jax
import numpy as np

from flax import struct
from flax import linen as nn
from jax import numpy as jnp
from jax.experimental import loops

from netket import jax as nkjax
from netket.hilbert import AbstractHilbert
from netket.utils import get_afun_if_module

PyTree = Any
PRNGKeyType = jnp.ndarray
SeedType = Union[int, PRNGKeyType]


@struct.dataclass
class SamplerState:
    """
    Base class holding the state of a sampler.
    """

    pass


@struct.dataclass
class Sampler(abc.ABC):
    """
    Base class for all samplers, containing the fields that all of them should posses.
    Note that fields marked with pytree_node=False are treated as static arguments
    when jitting.

    hilbert: The hilbert space to sample
    seed: The Initial state of the RNG.
    n_chains: The number of batches of the states to sample
    machine_pow: The power to which the machine should be exponentiated to generate the pdf.
    dtype: The dtype of the statees sampled
    """

    hilbert: AbstractHilbert = struct.field(pytree_node=False)
    """Hilbert space to be sampled."""
    ##: the fields below are present, but are created 'on the fly'
    # n_chains : int = struct.field(pytree_node = False, default = 8)
    """Number of batches along the chain"""
    # machine_pow : int = struct.field(default = 2)
    """Exponent of the pdf sampled"""

    def __post_init__(self):
        # Raise errors if hilbert is not an Hilbert
        if not isinstance(self.hilbert, AbstractHilbert):
            raise TypeError(
                "hilbert must be a subtype of netket.hilbert.AbstractHilbert, "
                + "instead, type {} is not.".format(type(self.hilbert))
            )

        if not isinstance(self.n_chains, int):
            raise TypeError("n_chains must be an integer")

    def init_state(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        seed: Optional[SeedType] = None,
    ) -> SamplerState:
        """
        Creates the structure holding the state of the sampler.
        """
        key = nkjax.PRNGKey(seed)

        return sampler._init_state(
            get_afun_if_module(machine), parameters, nkjax.mpi_split(key)
        )

    def reset(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        state: Optional[SamplerState] = None,
    ) -> SamplerState:
        """
        Resets the state of the sampler. To be used every time
        the parameters are changed.
        """
        if state is None:
            state = sampler_state(sampler, machine, parameters)

        return sampler._reset(get_afun_if_module(machine), parameters, state)

    def sample_next(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        state: Optional[SamplerState] = None,
    ) -> Tuple[jnp.ndarray, SamplerState]:
        """
        Samples the next state in the markov chain
        """
        if state is None:
            state = sampler_state(sampler, machine, parameters)

        return sampler._sample_next(get_afun_if_module(machine), parameters, state)

    def sample(sampler, *args, **kwargs) -> Tuple[jnp.ndarray, SamplerState]:
        return sample(sampler, *args, **kwargs)

    @partial(jax.jit, static_argnums=(1, 4))
    def _sample_chain(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        state: SamplerState,
        chain_length: int,
    ) -> Tuple[jnp.ndarray, SamplerState]:
        _sample_next = lambda state, _: sampler.sample_next(machine, parameters, state)

        state, samples = jax.lax.scan(
            _sample_next,
            state,
            xs=None,
            length=chain_length,
        )

        return samples, state

    @abc.abstractmethod
    def _init_state(sampler, machine, params, seed) -> SamplerState:
        raise NotImplementedError("init_state Not Implemented")

    @abc.abstractmethod
    def _reset(sampler, machine, parameters, state):
        raise NotImplementedError("reset Not Implemented")

    @abc.abstractmethod
    def _sample_next(sampler, machine, parameters, state=None):
        raise NotImplementedError("sample_next Not Implemented")


def sampler(clz_name):
    """
    Decorator to be used when defining samplers.
    Mainly responsible to define the three 'default' fields
     - n_chains
     - machine_pow.
     - dtype

    The reason for this method is that with python's dataclasses it is not possible to
    put fields with default values in a base class later inherited. So we use this trick
    and define them on the top class.
    """

    # Note: to define a field in a class we need two entries:
    #  - __dict__[__annotations__][fieldname] = fieldtype ,  which is only the type annotation
    #  - __dict__[fieldname] = default_value, which is modified through setattr.

    def sampler_decorator(clz):
        clz.__name__ = clz_name
        clz.__qualname__ = clz_name
        ann = clz.__dict__.get("__annotations__", {})

        # Error if those fields are already defined
        for name, type in ann.items():
            if (
                name == "hilbert"
                or name == "n_chains"
                or name == "machine_pow"
                or name == "dtype"
            ):
                raise TypeError(
                    "cannot define a type with field hilbert, n_chains, machine_pow or dtype as those"
                    + "are defined in the base Sampler class."
                )

        # n_chains : int = struct.field(pytree_node = False, default = 8)
        ann["n_chains"] = int
        setattr(clz, "n_chains", struct.field(pytree_node=False, default=8))

        # machine_pow : int = 2
        ann["machine_pow"] = int
        setattr(clz, "machine_pow", 2)

        ann["dtype"] = Any
        setattr(clz, "dtype", struct.field(pytree_node=False, default=np.float32))

        # Store the modified annotations
        setattr(clz, "__annotations__", ann)

        return struct.dataclass(clz)

    return sampler_decorator


def sampler_state(
    sampler: Sampler, machine: Union[Callable, nn.Module], parameters: PyTree
) -> SamplerState:
    """
    Constructs the state for the sampler. Dispatch on the
    sampler type.
    Args:
        sampler:
        machine:
        parameters:
    Returns:
        state : The SamplerState corresponding to this sampler
    """
    return sampler.init_state(machine, parameters)


def reset(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    state: Optional[SamplerState] = None,
) -> SamplerState:
    """
    Resets the state of a sampler.
    To be used after, for example, parameters have changed.
    If the state is not passed, then it is constructed
    Args:
        sampler:
        machine:
        parameters:
        state: the current state. If None, then it is constructed
    Returns:
        state : The SamplerState corresponding to this sampler

    """
    sampler.reset(machine, parameters, state)


def sample_next(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    state: Optional[SamplerState] = None,
) -> Tuple[jnp.ndarray, SamplerState]:
    """
    Samples the next state in the chain.

    Args:
        sampler:
        machine:
        parameters:
        state: the current state. If None, then it is constructed
    Returns:
        samples: a batch of samples
        state : The SamplerState corresponding to this sampler

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
        sampler: The sampler
        machine: The model or apply_fun to sample from (if it's a function it should have
            the signature f(parameters, σ) -> jnp.ndarray).
        parameters: The PyTree of parameters of the model.
        state: current state of the sampler. If None, then initialises it.
        chain_length: (default=1), the length of the chains.
    """
    if state is None:
        state = sampler.reset(machine, parameters, state)

    return sampler._sample_chain(machine, parameters, state, chain_length)


def samples(
    sampler: Sampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    *,
    state: Optional[SamplerState] = None,
    chain_length: int = 1,
):
    if state is None:
        state = sampler.reset(machine, parameters, state)

    for i in range(chain_length):
        samples, state = sampler._sample_chain(machine, parameters, state, 1)
        yield samples[0, :, :]
