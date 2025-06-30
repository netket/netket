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
import numpy as np
from math import prod

from netket import jax as nkjax
from netket.utils import get_afun_if_module, mpi
from netket.utils.types import Array, PyTree
from netket.hilbert import DiscreteHilbert, DoubledHilbert

from netket.utils.deprecation import deprecated


def split_array_mpi(array: Array) -> Array:
    """
    Splits the first dimension of the input array among mpi processes.
    Works like `mpi.scatter`, but assumes that the input array is available and
    identical on all ranks.
    !!! Warn
         The output is a numpy array.
    !!! Warn
         This should not be used with sharding (netket.netket_experimental_sharding=True)
    Args:
         array: A nd-array

    Result:
        A numpy array, of potentially different state on every mpi rank.
    """

    if mpi.n_nodes > 1:
        n_states = array.shape[0]
        states_n = np.arange(n_states)

        # divide the hilbert space in chunks for each node
        states_per_rank = np.array_split(states_n, mpi.n_nodes)

        return array[states_per_rank[mpi.rank]]
    else:
        return array


def to_array(
    hilbert: DiscreteHilbert,
    apply_fun: Callable[[PyTree, Array], Array],
    variables: PyTree,
    *,
    normalize: bool = True,
    allgather: bool = True,
    chunk_size: int | None = None,
    parallel_compute_axes=True,
) -> Array:
    """
    Computes `apply_fun(variables, states)` on all states of `hilbert` and returns
      the results as a vector.

    Args:
        normalize: If True, the vector is normalized to have L2-norm 1.
        allgather:
            When running with MPI:
                If True, the final wave function is stored in full at all MPI ranks.
            When running with netket_experimental_sharding=True:
                If allgather=True, the final wave function is a fully replicated array
                If allgather=False, the final wave function is a sharded array, padded
                with zeros to the next multiple of the number of devices
        chunk_size: Optional integer to specify the largest chunks of samples that
            the model will be evaluated upon. By default it is `None`, and when specified
            samples are split into chunks of at most `chunk_size`.

    Returns:

    """
    if not hilbert.is_indexable:
        raise RuntimeError("The hilbert space is not indexable")
    if parallel_compute_axes is False or jax.sharding.get_abstract_mesh().empty:
        parallel_compute_axes = None
    elif parallel_compute_axes is True:
        parallel_compute_axes = jax.sharding.get_abstract_mesh().axis_names
    elif isinstance(parallel_compute_axes, jax.P):
        parallel_compute_axes = tuple(parallel_compute_axes)

    if jax.sharding.get_abstract_mesh().empty:
        allgather = False

    apply_fun = get_afun_if_module(apply_fun)

    xs = hilbert.all_states()
    n_states = hilbert.n_states
    if parallel_compute_axes is not None:
        xs = nkjax.sharding.pad_axis_for_sharding(
            xs, axis=0, axis_name=parallel_compute_axes
        )
        xs = jax.sharding.reshard(xs, jax.P(parallel_compute_axes))

    psi = _to_array_rank(
        apply_fun,
        variables,
        xs,
        n_states,
        normalize,
        allgather,
        chunk_size,
    )

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
):
    """
    Computes apply_fun(variables, σ_rank) and gathers all results across all ranks.
    The input σ_rank should be a slice of all states in the hilbert space of equal
    length across all ranks because mpi4jax does not support allgatherv (yet).

    Args:
        n_states: total number of elements in the hilbert space.
    """

    # number of 'fake' states, in the last rank.
    n_fake_states = σ_rank.shape[0] * mpi.n_nodes - n_states

    log_psi_local = nkjax.lax.apply(
        partial(apply_fun, variables), σ_rank, batch_size=chunk_size
    )

    # last rank, get rid of fake elements
    if mpi.rank == mpi.n_nodes - 1 and n_fake_states > 0:
        log_psi_local = log_psi_local.at[-n_fake_states:].set(-jnp.inf)

    if normalize:
        # subtract logmax for better numerical stability
        logmax, _ = mpi.mpi_max_jax(log_psi_local.real.max())
        log_psi_local -= logmax

    psi_local = jnp.exp(log_psi_local)

    if normalize:
        # compute normalization
        norm2 = jnp.linalg.norm(psi_local) ** 2
        norm2, _ = mpi.mpi_sum_jax(norm2)
        psi_local /= jnp.sqrt(norm2)

    if allgather:
        psi = jax.sharding.reshard(psi_local, jax.P(None))
        return psi[:n_states]
    return psi_local


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


@deprecated(
    "The function `netket.nn.states_to_numbers` is deprecated. "
    "Please call `DiscreteHilbert.states_to_numbers` directly."
)
def states_to_numbers(hilbert: DiscreteHilbert, σ: Array) -> Array:
    return hilbert.states_to_numbers(σ)
