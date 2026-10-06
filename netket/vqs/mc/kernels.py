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

"""
This module implements some common kernels used by MCState and MCMixedState.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from netket.utils.types import PyTree, Array
import netket.jax as nkjax
from netket.operator import DiscreteJaxOperator
from netket.jax.sharding import sharding_decorator


def batch_discrete_kernel(kernel):
    """
    Batch a kernel that only works with 1 sample so that it works with a
    batch of samples.

    Works only for discrete-kernels who take two args as inputs
    """

    def vmapped_kernel(logpsi, pars, σ, args):
        """
        local_value kernel for MCState and generic operators
        """
        σp, mels = args

        if jnp.ndim(σp) != 3:
            σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
            mels = mels.reshape(σp.shape[:-1])

        vkernel = jax.vmap(kernel, in_axes=(None, None, 0, (0, 0)), out_axes=0)
        return vkernel(logpsi, pars, σ, (σp, mels))

    return vmapped_kernel


@batch_discrete_kernel
def local_value_kernel(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    """
    local_value kernel for MCState and generic operators
    """
    σp, mel = args
    return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))


def local_value_kernel_jax(
    logpsi: Callable, pars: PyTree, σ: Array, O: DiscreteJaxOperator
):
    """
    local_value kernel for MCState for jax-compatible operators
    """
    σp, mel = O.get_conn_padded(σ)
    logpsi_σ = logpsi(pars, σ)
    logpsi_σp = logpsi(pars, σp.reshape(-1, σp.shape[-1])).reshape(σp.shape[:-1])
    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


def local_value_kernel_jax_conn_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    chunk_size: int,
):
    """
    local_value kernel for MCState for jax-compatible operators
    """
    # IMPORTANT: pars must be passed as explicit arg (not captured in lambda) so that
    # shard_map's pvary/pcast mechanism can give it Manual sharding inside shard_map.
    apply_conn = nkjax.apply_chunked(logpsi, in_axes=(None, 0), chunk_size=chunk_size)

    σp, mel = O.get_conn_padded(σ)

    logpsi_σ = apply_conn(pars, σ)
    logpsi_σp = apply_conn(pars, σp.reshape(-1, σ.shape[-1])).reshape(σp.shape[:-1])

    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


def local_value_squared_kernel(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    """
    local_value kernel for MCState and Squared (generic) operators
    """
    return jnp.abs(local_value_kernel(logpsi, pars, σ, args)) ** 2


@batch_discrete_kernel
def local_value_op_op_cost(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    """
    local_value kernel for MCMixedState and generic operators
    """
    σp, mel = args

    σ_σp = jax.vmap(lambda σp, σ: jnp.hstack((σp, σ)), in_axes=(0, None))(σp, σ)
    σ_σ = jnp.hstack((σ, σ))
    return jnp.sum(mel * jnp.exp(logpsi(pars, σ_σp) - logpsi(pars, σ_σ)))


## Chunked versions of those kernels are defined below.


def local_value_kernel_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: int | None = None,
):
    """
    local_value kernel for MCState and generic operators
    """
    σp, mels = args

    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
        mels = mels.reshape(σp.shape[:-1])

    # IMPORTANT: pars must be passed as explicit arg (not captured in partial) so that
    # shard_map's pvary/pcast mechanism can give it Manual sharding inside shard_map.
    logpsi_chunked = nkjax.vmap_chunked(
        logpsi, in_axes=(None, 0), chunk_size=chunk_size
    )
    N = σ.shape[-1]

    logpsi_σ = logpsi_chunked(pars, σ.reshape((-1, N))).reshape(σ.shape[:-1] + (1,))
    logpsi_σp = logpsi_chunked(pars, σp.reshape((-1, N))).reshape(σp.shape[:-1])

    return jnp.sum(mels * jnp.exp(logpsi_σp - logpsi_σ), axis=-1)


def local_value_squared_kernel_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: int | None = None,
):
    """
    local_value kernel for MCState and Squared (generic) operators
    """
    return (
        jnp.abs(
            local_value_kernel_chunked(logpsi, pars, σ, args, chunk_size=chunk_size)
        )
        ** 2
    )


def local_value_op_op_cost_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    args: PyTree,
    *,
    chunk_size: int | None = None,
):
    """
    local_value kernel for MCMixedState and generic operators
    """
    σp, mels = args

    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
        mels = mels.reshape(σp.shape[:-1])

    σ_σp = jax.vmap(
        lambda σpi, σi: jax.vmap(lambda σp, σ: jnp.hstack((σp, σ)), in_axes=(0, None))(
            σpi, σi
        ),
        in_axes=(0, 0),
        out_axes=0,
    )(σp, σ)
    σ_σ = jax.vmap(lambda σi: jnp.hstack((σi, σi)), in_axes=0)(σ)

    return local_value_kernel_chunked(
        logpsi, pars, σ_σ, (σ_σp, mels), chunk_size=chunk_size
    )


def local_value_kernel_jax_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    *,
    chunk_size: int | None = None,
):
    """
    local_value kernel for MCState and jaxcoompatible operators
    """
    if chunk_size >= O.max_conn_size:
        # IMPORTANT: pars must be passed as explicit arg (not captured in lambda) so that
        # shard_map's pvary/pcast mechanism can give it Manual sharding inside shard_map.
        def _local_value_kernel(pars, s, O):
            return local_value_kernel_jax(logpsi, pars, s, O)

        local_value_chunked = nkjax.apply_chunked(
            _local_value_kernel,
            in_axes=(None, 0, None),
            chunk_size=max(1, chunk_size // max(1, O.max_conn_size)),
            pvary_argnums=(0,),
        )
        return local_value_chunked(pars, σ, O)
    else:
        return local_value_kernel_jax_conn_chunked(logpsi, pars, σ, O, chunk_size)


## Flattened kernel for jax operators, skipping the zero connected elements.

_FLATTENED_N_CHUNKS_UNCHUNKED = 16
"""Number of chunks the padded connected configurations are split into by
:func:`local_value_kernel_jax_flattened` when no ``chunk_size`` is given."""


def local_value_kernel_jax_flattened(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    *,
    chunk_size: int | None = None,
):
    r"""
    local_value kernel for MCState and jax-compatible operators that evaluates
    :math:`\log\psi` only on the connected configurations :math:`x' \neq x`.

    :func:`local_value_kernel_jax` evaluates the network on all
    ``n_samples × max_conn_size`` configurations returned by
    :meth:`~netket.operator.DiscreteJaxOperator.get_conn_padded`, including
    the diagonal :math:`x' = x` and the padding entries, which NetKet's
    operators fill with :math:`x` and a zero matrix element. This kernel
    instead, on every device:

    - uses :math:`\log\psi(x)` for the connected configurations equal to
      :math:`x`, without evaluating the network;
    - compacts the flattened mask of the other ones with
      :func:`jax.numpy.nonzero` into an index buffer of static size
      ``n_local_samples × max_conn_size``;
    - evaluates :math:`\log\psi` on the selected configurations in chunks of
      ``chunk_size`` configurations (``n_local_samples × max_conn_size / 16``
      if ``chunk_size`` is None). The number of chunks depends on the number
      of selected configurations. Full chunks have the requested size; the
      remainder is split into power-of-two batches, down to a single row.
      No extra rows are evaluated to fill a chunk. All batch shapes are
      static, so a varying number of connections never triggers a
      recompilation;
    - gathers the values back to the padded layout and computes the local
      values with the same summation as :func:`local_value_kernel_jax`.

    The selection only depends on the configurations, not on the values of
    the matrix elements: a connected configuration whose matrix element is
    zero for the current coefficients of the operator is still evaluated, so
    that the derivatives with respect to those coefficients are correct.

    The forward reduction does not introduce an atomic scatter-add. Repeated
    results still depend on the determinism of the model and backend; changing
    the network batch shape can change floating-point rounding.

    The kernel runs per device (inside :func:`jax.shard_map` over the sample
    axis), because the compaction is a global operation that GSPMD would
    otherwise replicate on every device.

    Full chunks are evaluated in blocks with static trip counts, avoiding a
    data-dependent GPU loop for every network batch. Runtime conditions select
    the occupied blocks without evaluating empty chunks. When differentiated
    (for example by :func:`~netket.vqs.expect_and_grad` with
    ``use_covariance=False``), a custom JVP uses a :func:`jax.lax.scan` over the
    maximum number of chunks, skipping empty ones with :func:`jax.lax.cond`.
    This supports forward, reverse, and higher-order differentiation.

    The result agrees with :func:`local_value_kernel_jax` up to the rounding
    of :math:`\log\psi`, which can depend on the batch it is evaluated in.

    Args:
        logpsi: the log-amplitude function.
        pars: the variables of the model.
        σ: the samples, of shape ``(n_samples, hilbert.size)``.
        O: the operator.
        chunk_size: the number of connected configurations on which the
            network is evaluated at once.
    """
    kernel = nkjax.HashablePartial(
        _local_value_kernel_jax_flattened, logpsi, chunk_size=chunk_size
    )
    # IMPORTANT: pars must be passed as explicit arg (not captured in the partial)
    # so that shard_map's pvary/pcast mechanism can give it Manual sharding.
    return sharding_decorator(
        kernel,
        sharded_args_tree=(False, True, False),
        pvary_args_tree=(True, False, False),
    )(pars, σ, O)


def _local_value_kernel_jax_flattened(logpsi, pars, σ, O, *, chunk_size):
    # Runs on the samples of a single device.
    n_samples, N = σ.shape
    σp, mels = O.get_conn_padded(σ)
    max_conn_size = mels.shape[-1]
    n_conns = n_samples * max_conn_size

    if n_conns == 0:
        # No connected elements (e.g. an empty operator): the local values are
        # zero, and the loop below cannot be traced on an empty buffer. The
        # zeros are built from σ so that they are varying inside shard_map.
        dtype = jnp.result_type(mels.dtype, jax.eval_shape(logpsi, pars, σ).dtype)
        return jnp.zeros_like(σ[:, 0], dtype=dtype)

    # The connected configurations equal to σ (the diagonal and the padding)
    # are found by comparison, so that we do not rely on the operator placing
    # them at a given position.
    is_conn = jnp.any(σp != jnp.expand_dims(σ, -2), axis=-1).reshape(-1)

    if chunk_size is None:
        chunk_size = -(-n_conns // _FLATTENED_N_CHUNKS_UNCHUNKED)
    chunk_size = max(1, min(chunk_size, n_conns))
    n_chunks_max = -(-n_conns // chunk_size)
    buffer_size = n_chunks_max * chunk_size

    # Indices of the selected configurations, sample-major. The filling only
    # needs to be a valid configuration.
    n_conn = is_conn.sum()
    (idx,) = jnp.nonzero(is_conn, size=buffer_size, fill_value=n_conns - 1)

    logpsi_σ = nkjax.apply_chunked(
        logpsi, in_axes=(None, 0), chunk_size=chunk_size, axis_0_is_sharded=False
    )(pars, σ)
    logpsi_conn = _flattened_logpsi(
        logpsi, chunk_size, pars, σp.reshape(-1, N), idx, n_conn
    )

    # Gather the values back to the padded layout, from the position of every
    # selected configuration in the compact buffer. No terms are accumulated in
    # the forward pass, and in the backward pass (a scatter-add) every entry
    # receives at most one nonzero term, so the result does not depend on the
    # order of the updates. (A scatter of complex numbers is also very slow on
    # GPU.)
    position = jnp.maximum(jnp.cumsum(is_conn) - 1, 0)
    logpsi_σp = jnp.where(
        is_conn, logpsi_conn[position], jnp.repeat(logpsi_σ, max_conn_size)
    ).reshape(n_samples, max_conn_size)
    return jnp.sum(mels * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)


def _flattened_logpsi_chunk(logpsi, chunk_size, pars, σp, idx, start):
    # Selected rows are packed once before the loop. A contiguous slice avoids
    # repeating an indirect gather at every network invocation.
    return logpsi(pars, jax.lax.dynamic_slice_in_dim(σp, start, chunk_size))


def _flattened_zeros(logpsi, pars, σp, idx):
    # Built from the data so that, inside shard_map, it is varying over the
    # sample axis like the loop updates.
    dtype = jax.eval_shape(logpsi, pars, σp[:1]).dtype
    return jnp.zeros_like(idx, dtype=dtype)


def _flattened_logpsi_tail(logpsi, chunk_size, pars, σp, idx, n_conn, out):
    """Evaluate the remainder exactly, using statically sized batches.

    A static set of conditional branches covers every possible remainder.
    Each active branch evaluates only occupied rows. Conditions run inside
    shard_map, so devices with different counts can take different branches.
    """
    start = (n_conn // chunk_size) * chunk_size
    for bit in reversed(range((chunk_size - 1).bit_length())):
        size = 1 << bit

        def evaluate(state):
            out, start = state
            values = _flattened_logpsi_chunk(logpsi, size, pars, σp, idx, start)
            return (
                jax.lax.dynamic_update_slice_in_dim(out, values, start, axis=0),
                start + size,
            )

        out, start = jax.lax.cond(
            n_conn - start >= size, evaluate, lambda state: state, (out, start)
        )
    return out


def _flattened_logpsi_blocks(logpsi, chunk_size, pars, σp, idx, n_conn):
    # GPU execution of a dynamic-trip-count loop can cost more than the model.
    # Decompose the occupied full chunks into powers of two: every active block
    # has a static trip count, with only logarithmically many runtime decisions.
    n_chunks = n_conn // chunk_size
    n_chunks_max = idx.shape[0] // chunk_size
    out = _flattened_zeros(logpsi, pars, σp, idx)

    def evaluate_blocks(out):
        start = jnp.zeros_like(n_conn)
        for bit in reversed(range(n_chunks_max.bit_length())):
            length = 1 << bit

            def evaluate(state):
                out, start = state

                def body(c, out):
                    offset = (start + c) * chunk_size
                    values = _flattened_logpsi_chunk(
                        logpsi, chunk_size, pars, σp, idx, offset
                    )
                    return jax.lax.dynamic_update_slice_in_dim(
                        out, values, offset, axis=0
                    )

                out = jax.lax.fori_loop(0, length, body, out)
                return out, start + length

            out, start = jax.lax.cond(
                n_chunks - start >= length,
                evaluate,
                lambda state: state,
                (out, start),
            )
        return out

    # No full chunks means no block-selection work is needed.
    out = jax.lax.cond(n_chunks > 0, evaluate_blocks, lambda out: out, out)
    return _flattened_logpsi_tail(logpsi, chunk_size, pars, σp, idx, n_conn, out)


def _flattened_logpsi_scan(logpsi, chunk_size, pars, σp, idx, n_conn):
    n_chunks_max = idx.shape[0] // chunk_size
    zeros = _flattened_zeros(logpsi, pars, σp, idx[:chunk_size])

    def body(_, c):
        out = jax.lax.cond(
            (c + 1) * chunk_size <= n_conn,
            lambda c: _flattened_logpsi_chunk(
                logpsi, chunk_size, pars, σp, idx, c * chunk_size
            ),
            lambda c: zeros,
            c,
        )
        return None, out

    _, out = jax.lax.scan(body, None, jnp.arange(n_chunks_max))
    return _flattened_logpsi_tail(
        logpsi, chunk_size, pars, σp, idx, n_conn, out.reshape(-1)
    )


def _flattened_logpsi(logpsi, chunk_size, pars, σp, idx, n_conn):
    # The primal uses fixed-length blocks. For AD, a fixed-capacity scan keeps
    # the derivative graph independent of the combination of active blocks.
    @jax.custom_jvp
    def flattened_logpsi(*args):
        return _flattened_logpsi_blocks(logpsi, chunk_size, *args)

    @flattened_logpsi.defjvp
    def flattened_logpsi_jvp(primals, tangents):
        return jax.jvp(
            lambda *args: _flattened_logpsi_scan(logpsi, chunk_size, *args),
            primals,
            tangents,
        )

    # Packing fixed-capacity storage does not evaluate any padded model rows.
    # Only the first n_conn entries reach the full chunks or exact tail.
    return flattened_logpsi(pars, σp[idx], idx, n_conn)


def local_value_kernel_jax_unique(logpsi, pars, σ, O, *, chunk_size=None):
    """Local values with one model evaluation per distinct configuration.

    References and connected configurations are deduplicated together on each
    device. Matrix elements and their original row-wise reduction are retained.
    Padding equal to a reference shares its model value; off-diagonal entries
    with currently zero coefficients remain available for coefficient AD.

    Exact comparisons, rather than hashes, identify equal configurations.
    Full chunks and exact tails use the shared flattened evaluator. Expansion
    uses a segmented scan so its transpose does not require an atomic sum for
    repeated indices. Models must be pointwise in their input configurations;
    their compute precision still determines numerical accuracy.
    """
    kernel = nkjax.HashablePartial(
        _local_value_kernel_jax_unique, logpsi, chunk_size=chunk_size
    )
    return sharding_decorator(
        kernel,
        sharded_args_tree=(False, True, False),
        pvary_args_tree=(True, False, False),
    )(pars, σ, O)


def _unique_row_plan(rows):
    """Exact lexicographic grouping, with fixed-capacity representative indices."""
    order = jnp.lexsort(rows.T[::-1])
    return _group_sorted_rows(rows, order)


def _group_sorted_rows(rows, order):
    size = rows.shape[0]
    sorted_rows = rows[order]
    starts = jnp.concatenate(
        (jnp.ones(1, dtype=bool), jnp.any(sorted_rows[1:] != sorted_rows[:-1], axis=1))
    )
    count = starts.sum()
    (positions,) = jnp.nonzero(starts, size=size, fill_value=size)
    representatives = order[jnp.minimum(positions, size - 1)]
    # Every out-of-bounds destination is distinct too, so the unique_indices
    # promise in the expansion is true even for unused capacity.
    positions = jnp.where(jnp.arange(size) < count, positions, size + jnp.arange(size))
    return order, starts, positions, representatives, count


def _mix_uint32(value):
    value = (value ^ (value >> 16)) * jnp.uint32(0x7FEB352D)
    value = (value ^ (value >> 15)) * jnp.uint32(0x846CA68B)
    return value ^ (value >> 16)


def _row_fingerprints(rows):
    """Cheap candidate keys; equality is always checked on the full rows."""
    if jnp.issubdtype(rows.dtype, jnp.complexfloating):
        rows = jnp.concatenate((rows.real, rows.imag), axis=-1)
    if jnp.issubdtype(rows.dtype, jnp.floating):
        if rows.dtype.itemsize < 4:
            rows = rows.astype(jnp.float32)
        bits = jax.lax.bitcast_convert_type(
            rows, jnp.uint64 if rows.dtype.itemsize == 8 else jnp.uint32
        )
    else:
        bits = rows.astype(jnp.uint64 if rows.dtype.itemsize == 8 else jnp.uint32)
    if bits.dtype.itemsize == 8:
        bits = bits ^ (bits >> 32)
    bits = bits.astype(jnp.uint32)
    weights = _mix_uint32(jnp.arange(rows.shape[-1], dtype=jnp.uint32) + 1) | 1
    return jnp.sum(_mix_uint32(bits) * weights, axis=-1, dtype=jnp.uint32)


def _fingerprint_row_plan(rows, fingerprints=None):
    # A collision can separate occurrences of one configuration. Those rows
    # may then be evaluated more than once, but distinct rows never share a
    # model output because _group_sorted_rows checks exact equality.
    if fingerprints is None:
        fingerprints = _row_fingerprints(rows)
    return _group_sorted_rows(rows, jnp.argsort(fingerprints, stable=True))


def _sample_reuse_predicate(samples):
    # This is a fixed data heuristic, not a timing tuner. Checking B references
    # costs much less than sorting B * max_conn_size connected configurations.
    # Require at least 1/8 repeated samples before trying connected-state reuse.
    # Shared neighbours of distinct samples can be missed intentionally.
    order = jnp.argsort(_row_fingerprints(samples), stable=True)
    ordered = samples[order]
    repeated = jnp.all(ordered[1:] == ordered[:-1], axis=-1).sum()
    return 8 * repeated >= samples.shape[0]


def local_value_kernel_jax_fingerprint(logpsi, pars, σ, O, *, chunk_size=None):
    """Reuse exactly matching rows found by inexpensive fingerprint grouping.

    Collisions can cause some duplicate evaluations, but unequal
    configurations never share a model value. No sample gate or tuner runs.
    """
    return local_value_kernel_jax_reuse(
        logpsi, pars, σ, O, chunk_size=chunk_size, check_samples=False
    )


def local_value_kernel_jax_reuse(
    logpsi, pars, σ, O, *, chunk_size=None, check_samples=True
):
    """Heuristic reuse with exact equality checks and a compact fallback.

    A cheap reference-sample check selects fingerprint grouping only when at
    least 1/8 of the samples are repeated. Otherwise use ordinary compaction.
    This can miss savings but cannot merge unequal configurations. Set
    check_samples=False to benchmark fingerprint grouping without the gate.
    The choice is made per device at runtime without recompiling.
    """
    kernel = nkjax.HashablePartial(
        _local_value_kernel_jax_reuse,
        logpsi,
        chunk_size=chunk_size,
        check_samples=check_samples,
    )
    return sharding_decorator(
        kernel,
        sharded_args_tree=(False, True, False),
        pvary_args_tree=(True, False, False),
    )(pars, σ, O)


def _local_value_kernel_jax_reuse(logpsi, pars, σ, O, *, chunk_size, check_samples):
    def reuse(args):
        return _local_value_kernel_jax_unique(
            logpsi, *args, chunk_size=chunk_size, row_plan=_fingerprint_row_plan
        )

    if not check_samples:
        return reuse((pars, σ, O))
    return jax.lax.cond(
        _sample_reuse_predicate(σ),
        reuse,
        lambda args: _local_value_kernel_jax_flattened(
            logpsi, *args, chunk_size=chunk_size
        ),
        (pars, σ, O),
    )


def _expand_unique_rows(values, order, starts, positions):
    """Repeat unique values with a deterministic, differentiable transpose.

    Deposit once at each group start, then propagate through the group. AD of
    this scan sums repeated cotangents in a fixed tree, avoiding the overlapping
    scatter-add generated by a direct gather with repeated indices.
    """
    # XLA's GPU lowering of complex scatters can be orders of magnitude slower
    # than the two real scatters, even when the destinations are unique.
    # Splitting is exact and gives the same linear map and transpose.
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        return jax.lax.complex(
            _expand_unique_rows(values.real, order, starts, positions),
            _expand_unique_rows(values.imag, order, starts, positions),
        )
    seeds = (
        jnp.zeros_like(values)
        .at[positions]
        .set(values, mode="drop", unique_indices=True, indices_are_sorted=True)
    )

    def combine(left, right):
        left_start, left_value = left
        right_start, right_value = right
        return (
            left_start | right_start,
            jnp.where(right_start, right_value, left_value + right_value),
        )

    _, expanded = jax.lax.associative_scan(combine, (starts, seeds))
    return jnp.zeros_like(expanded).at[order].set(expanded, unique_indices=True)


def _local_value_kernel_jax_unique(
    logpsi, pars, σ, O, *, chunk_size, row_plan=_unique_row_plan
):
    n_samples, n_sites = σ.shape
    σp, mels = O.get_conn_padded(σ)
    if mels.size == 0:
        dtype = jnp.result_type(mels.dtype, jax.eval_shape(logpsi, pars, σ).dtype)
        return jnp.zeros_like(σ[:, 0], dtype=dtype)

    rows = jnp.concatenate((σ, σp.reshape(-1, n_sites)))
    order, starts, positions, representatives, count = row_plan(rows)
    if chunk_size is None:
        chunk_size = -(-rows.shape[0] // _FLATTENED_N_CHUNKS_UNCHUNKED)
    chunk_size = max(1, min(chunk_size, rows.shape[0]))
    capacity = -(-rows.shape[0] // chunk_size) * chunk_size
    indices = jnp.pad(representatives, (0, capacity - rows.shape[0]))
    unique_values = _flattened_logpsi(logpsi, chunk_size, pars, rows, indices, count)
    expanded = _expand_unique_rows(
        unique_values[: rows.shape[0]], order, starts, positions
    )
    reference = expanded[:n_samples]
    connected = expanded[n_samples:].reshape(mels.shape)
    return jnp.sum(mels * jnp.exp(connected - reference[:, None]), axis=-1)
