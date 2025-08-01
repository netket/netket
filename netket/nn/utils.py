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

from functools import partial
from collections.abc import Callable
from typing import cast

import jax
from jax import numpy as jnp
from jax.core import concrete_or_error
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
from math import prod

from netket import jax as nkjax
from netket.utils import get_afun_if_module
from netket.utils.types import Array, PyTree
from netket.hilbert import DiscreteHilbert, DoubledHilbert

from netket.utils import config
from netket.jax.sharding import (
    extract_replicated,
    distribute_to_devices_along_axis,
)


def to_array(
    hilbert: DiscreteHilbert,
    apply_fun: Callable[[PyTree, Array], Array],
    variables: PyTree,
    *,
    normalize: bool = True,
    allgather: bool = True,
    chunk_size: int | None = None,
) -> Array:
    """
    Computes `apply_fun(variables, states)` on all states of `hilbert` and returns
    the results as a vector.

    Args:
        normalize: If True, the vector is normalized to have L2-norm 1.
        allgather:
            If allgather=True, the final wave function is a fully replicated array.
            If allgather=False, the final wave function is a sharded array, padded
            with zeros to the next multiple of the number of devices.
        chunk_size: Optional integer to specify the largest chunks of samples that
            the model will be evaluated upon. By default it is `None`, and when specified
            samples are split into chunks of at most `chunk_size`.

    Returns:
        Array: The computed array.
    """
    if not hilbert.is_indexable:
        raise RuntimeError("The hilbert space is not indexable")

    apply_fun = get_afun_if_module(apply_fun)

    if config.netket_experimental_sharding:  # type: ignore
        x = hilbert.all_states()
        xs, mask = distribute_to_devices_along_axis(x, pad=True, pad_value=x[0])
        n_states = hilbert.n_states
    else:
        xs = hilbert.all_states()
        mask = None
        n_states = xs.shape[0]

    psi = _to_array_rank(
        apply_fun,
        variables,
        xs,
        n_states,
        normalize,
        allgather,
        chunk_size,
        mask,
    )

    if allgather and config.netket_experimental_sharding:  # type: ignore
        psi = np.asarray(extract_replicated(psi))

    return psi


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
def _to_array_rank(
    apply_fun,
    variables,
    σ_rank,
    n_states,
    normalize,
    allgather,
    chunk_size,
    mask=None,
):
    """
    Computes apply_fun(variables, σ_rank) and gathers all results across all ranks.
    The input σ_rank can be sharded.

    Args:
        n_states: total number of elements in the hilbert space.
    """

    if chunk_size is not None:
        apply_fun = nkjax.apply_chunked(
            apply_fun, in_axes=(None, 0), chunk_size=chunk_size
        )

    # number of 'fake' states, in the last rank.
    n_fake_states = σ_rank.shape[0] - n_states

    log_psi_local = apply_fun(variables, σ_rank)

    # last rank, get rid of fake elements
    if n_fake_states > 0:
        log_psi_local = log_psi_local.at[-n_fake_states:].set(-jnp.inf)

    if normalize:
        # subtract logmax for better numerical stability
        log_psi_local -= log_psi_local.real.max()

    psi_local = jnp.exp(log_psi_local)

    if mask is not None:
        # when running under netket_experimental_sharding,
        # we pad the Hilbert space with extra fake entries,
        # which in here we mask out to 0
        psi_local = psi_local * mask

    if normalize:
        # compute normalization
        norm2 = jnp.linalg.norm(psi_local) ** 2
        psi_local /= jnp.sqrt(norm2)

    if allgather:
        psi = psi_local.reshape(-1)
    else:
        psi = psi_local

    # gather/replicate
    if allgather and config.netket_experimental_sharding:  # type: ignore
        sharding = NamedSharding(jax.sharding.get_abstract_mesh(), P())
        psi = jax.lax.with_sharding_constraint(psi, sharding)

    # remove fake states
    psi = psi[0:n_states]

    return psi


def to_matrix(
    hilbert: DoubledHilbert,
    machine: Callable[[PyTree, Array], Array],
    params: PyTree,
    *,
    normalize: bool = True,
    chunk_size: int | None = None,
) -> Array:
    if not hilbert.is_indexable:
        raise RuntimeError("The hilbert space is not indexable")

    psi = to_array(hilbert, machine, params, normalize=False, chunk_size=chunk_size)

    L = hilbert.physical.n_states
    rho = psi.reshape((L, L))
    if normalize:
        trace = jnp.trace(rho)
        rho /= trace

    return rho


def _get_output_idx(
    shape: tuple[int, ...], max_bits: int | None = None
) -> tuple[tuple[int, ...], int]:
    bits_per_local_occupation = tuple(np.ceil(np.log2(shape)).astype(int))
    if max_bits is None:
        max_bits = max(bits_per_local_occupation)
        max_bits = cast(int, max_bits)
    _output_idx: list[int] = []
    offset = 0
    for b in bits_per_local_occupation:
        _output_idx.extend([i + offset for i in range(b)][::-1])
        offset += max_bits
    output_idx = tuple(_output_idx)
    return output_idx, max_bits


def _separate_binary_indices(
    shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    binary_indices = tuple([i for i in range(len(shape)) if shape[i] == 2])
    non_binary_indices = tuple([i for i in range(len(shape)) if shape[i] != 2])
    return binary_indices, non_binary_indices


@partial(jax.jit, static_argnames=("hilbert", "max_bits"))
def binary_encoding(
    hilbert: DiscreteHilbert,
    x: Array,
    *,
    max_bits: int | None = None,
) -> Array:
    """
    Encodes the array `x` into a set of binary-encoded variables described by
    the shape of a Hilbert space. The i-th element of x will be encoded in
    {code}`ceil(log2(shape[i]))` bits.

    Args:
        hilbert: Hilbert space of the samples that are to be encoded.
        x: The array to encode.
        max_bits: The maximum number of bits to use for each element of `x`.
    """
    x = hilbert.states_to_local_indices(x)
    shape = tuple(hilbert.shape)
    concrete_or_error(None, shape, "Shape must be known statically")
    output_idx, max_bits = _get_output_idx(shape, max_bits)
    binarised_states = jnp.zeros(
        (
            *x.shape,
            max_bits,
        ),
        dtype=x.dtype,
    )
    binary_indices, non_binary_indices = _separate_binary_indices(shape)
    for i in non_binary_indices:
        substates = x[..., i].astype(int)[..., jnp.newaxis]
        binarised_states = (
            binarised_states.at[..., i, :]
            .set(
                substates & 2 ** jnp.arange(binarised_states.shape[-1], dtype=int) != 0
            )
            .astype(x.dtype)
        )
    for i in binary_indices:
        binarised_states = binarised_states.at[..., i, 0].set(x[..., i])
    return binarised_states.reshape(
        *binarised_states.shape[:-2], prod(binarised_states.shape[-2:])
    )[..., output_idx]
