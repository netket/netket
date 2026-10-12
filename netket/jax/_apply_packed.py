# Copyright 2026 The NetKet Authors - All rights reserved.
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

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap
from jax.custom_derivatives import SymbolicZero

from netket.utils.types import Array, PyTree
from netket.jax._compat import aval_varying_axes
from netket.jax.lax import pcast

_N_CHUNKS_UNCHUNKED = 16
"""Number of chunks the rows are split into by :func:`apply_packed` when no
``chunk_size`` is given."""


def apply_packed(
    f: Callable,
    pars: PyTree,
    x: PyTree,
    where: Array,
    fill: PyTree,
    *,
    chunk_size: int | None,
    min_chunk_size: int | None = None,
) -> PyTree:
    r"""
    Row-wise ``jnp.where(where, f(pars, x), fill)``, evaluating ``f`` only on
    the rows where ``where`` is True.

    The selected rows are packed into chunks of ``chunk_size`` rows, and the
    remaining ones into chunks of decreasing powers of two down to
    ``min_chunk_size``, the last of which is padded. The number of chunks
    depends on the number of selected rows, which can change from call to call
    without recompiling: only their shapes are static. The full chunks are
    evaluated in blocks of :math:`2^k` chunks, one for every bit of their
    number, and every smaller chunk at most once, each selected by a
    :func:`jax.lax.cond`, in loops with a static number of iterations. On GPU,
    a loop whose number of iterations depends on the data waits for the device
    at every iteration, while there are only logarithmically many conds.

    ``f`` must be row-wise: the rows of its output must only depend on the
    corresponding rows of its input. It is evaluated on a few chunk shapes,
    and traced once per shape.

    The function can be differentiated with respect to ``pars`` and ``fill``
    in forward and reverse mode, at any order. Like a :func:`jax.lax.scan`
    over all the chunks, the reverse mode stores the intermediate values of
    ``f`` for all the rows, also for the chunks that are skipped.

    Under :func:`jax.vmap` over ``x`` or ``where`` (the rows), the batch is
    folded into the rows and packed at once, so that only the selected rows of
    all the batch are evaluated, in chunks of ``chunk_size`` rows. A batch of
    ``pars`` or ``fill`` alone is evaluated with the same packing. When both
    the rows and ``pars`` are batched, the elements of the batch are evaluated
    one after the other. Under :func:`jax.vmap` of a derivative, every element
    of the batch is packed separately, and evaluates as many chunks as the
    largest one. In all cases the chunks are selected by :func:`jax.lax.cond`.

    The function runs on the arrays it is given: inside of
    :func:`jax.shard_map` it packs the rows of every device separately.

    Args:
        f: the function to evaluate, ``f(pars, x_rows)``.
        pars: the arguments of ``f`` that are the same for all rows.
        x: a pytree of arrays with ``n`` rows.
        where: the boolean mask of the rows on which ``f`` is evaluated, of
            shape ``(n,)``.
        fill: the values of the other rows, a pytree like the output of ``f``
            whose leaves are broadcastable to the output.
        chunk_size: the number of rows on which ``f`` is evaluated at once
            (``where.size / 16`` if None).
        min_chunk_size: the size of the smallest chunk (``chunk_size // 8`` if
            None, rounded down to a power of two). Every chunk size is compiled
            separately: ``min_chunk_size=1`` pads nothing,
            ``min_chunk_size=chunk_size`` compiles a single chunk size.

    Returns:
        A pytree like the output of ``f``, with ``n`` rows.
    """
    where = jnp.asarray(where, dtype=bool)
    if where.ndim != 1:
        raise ValueError(f"where must have 1 dimension, not {where.ndim}.")
    impl = partial(_apply_packed, jax.jit(f), chunk_size, min_chunk_size)

    batched = custom_vmap(impl)
    batched.def_vmap(
        partial(_apply_packed_vmap_rule, impl, f, chunk_size, min_chunk_size)
    )
    packed = jax.custom_jvp(batched)
    packed.defjvp(partial(_apply_packed_jvp, impl), symbolic_zeros=True)
    return packed(pars, x, where, fill)


def _apply_packed(f, chunk_size, min_chunk_size, pars, x, where, fill):
    shape = where.shape
    n_rows = where.size
    is_selected = where.reshape(-1)

    def evaluate(i):
        # f on the rows at the indices i.
        return f(pars, jax.tree.map(lambda a: a[i], x))

    out_struct = _row_struct(f, pars, x)
    fill = jax.tree.map(
        lambda v, s: jnp.broadcast_to(v, shape + s.shape[1:]).reshape(
            (n_rows,) + s.shape[1:]
        ),
        fill,
        out_struct,
    )
    if n_rows == 0:
        # Nothing to evaluate, and the chunks below cannot be traced on an
        # empty buffer.
        return jax.tree.map(
            lambda v, s: v.astype(jnp.result_type(v, s.dtype)).reshape(
                shape + s.shape[1:]
            ),
            fill,
            out_struct,
        )

    if chunk_size is None:
        chunk_size = -(-n_rows // _N_CHUNKS_UNCHUNKED)
    chunk_size = max(1, min(chunk_size, n_rows))
    if min_chunk_size is None:
        min_chunk_size = chunk_size // 8
    min_chunk_size = max(1, min(min_chunk_size, chunk_size))
    if min_chunk_size < chunk_size:
        min_chunk_size = 1 << (min_chunk_size.bit_length() - 1)
    n_chunks_max = -(-n_rows // chunk_size)
    # The padded last chunk can end after the last full chunk.
    buffer_size = n_chunks_max * chunk_size + min_chunk_size

    # Indices of the selected rows. The filling only needs to be a valid row.
    n_selected = is_selected.sum()
    (idx,) = jnp.nonzero(is_selected, size=buffer_size, fill_value=n_rows - 1)
    # Inside shard_map, the buffers must vary over the manual axes like the
    # values written into them.
    axes = set()
    for leaf in jax.tree.leaves((pars, x, where, fill)):
        axes.update(aval_varying_axes(leaf))
    out = _evaluate_packed(
        evaluate,
        out_struct,
        chunk_size,
        min_chunk_size,
        n_chunks_max,
        idx,
        n_selected,
        tuple(sorted(axes)),
    )

    # Gather the values back to the rows, from the position of every selected
    # row in the packed buffer. No terms are accumulated in the forward pass,
    # and in the backward pass (a scatter-add) every entry receives at most one
    # nonzero term, so the result does not depend on the order of the updates.
    # (A scatter of complex numbers is also very slow on GPU.)
    position = jnp.maximum(jnp.cumsum(is_selected) - 1, 0)

    def unpack(o, v):
        mask = is_selected.reshape((n_rows,) + (1,) * (o.ndim - 1))
        return jnp.where(mask, o[position], v).reshape(shape + o.shape[1:])

    return jax.tree.map(unpack, out, fill)


def _vary(x, axes):
    # x, varying over the manual axes `axes` of the enclosing shard_map.
    missing = tuple(a for a in axes if a not in aval_varying_axes(x))
    return pcast(x, missing, to="varying") if missing else x


def _row_struct(f, pars, x):
    # The shapes and dtypes of the output of f on a single row.
    row = jax.tree.map(lambda a: jax.ShapeDtypeStruct((1,) + a.shape[1:], a.dtype), x)
    return jax.eval_shape(f, pars, row)


def _evaluate_packed(
    evaluate,
    out_struct,
    chunk_size,
    min_chunk_size,
    n_chunks_max,
    idx,
    n_selected,
    varying_axes,
):
    # evaluate on the first n_selected packed indices idx. The entries after
    # them are zero or hold the values of the padding of the last chunk.
    out = jax.tree.map(
        lambda s: _vary(
            jnp.zeros((idx.shape[0],) + s.shape[1:], s.dtype), varying_axes
        ),
        out_struct,
    )

    def evaluate_chunks(size, n_chunks, out, start):
        # n_chunks consecutive chunks of size rows, from start.
        def body(c, out):
            offset = start + c * size
            i = jax.lax.dynamic_slice_in_dim(idx, offset, size)
            return jax.tree.map(
                lambda o, v: jax.lax.dynamic_update_slice_in_dim(o, v, offset, axis=0),
                out,
                evaluate(i),
            )

        if n_chunks == 1:
            return body(0, out)
        return jax.lax.fori_loop(0, n_chunks, body, out)

    def evaluate_if(pred, size, n_chunks, out, start):
        # Like the buffers, the start varies over the manual axes, also when it
        # is a constant: a batched predicate turns the cond into a select, which
        # would otherwise add values of different manual axis types.
        start = _vary(jnp.asarray(start, dtype=n_selected.dtype), varying_axes)
        return jax.lax.cond(
            pred,
            partial(evaluate_chunks, size, n_chunks),
            lambda out, start: out,
            out,
            start,
        )

    # Under vmap, a batched predicate would turn the conds into selects, which
    # evaluate every chunk (and that reverse mode cannot transpose when the
    # derivative is batched): every element of the batch evaluates as many
    # chunks as the largest one instead.
    n_selected = _max_over_batch(n_selected)

    # The full chunks, in blocks of 2^k chunks for every bit k of their number.
    # Blocks of 1, 2, ..., 2^(t-1) chunks cover up to 2^t - 1 chunks, and the
    # first block covers the ones above, so that the blocks hold n_chunks_max
    # chunks in total: the reverse-mode derivative stores the values of every
    # block, also of those that are skipped.
    n_full = n_selected // chunk_size
    t = (n_chunks_max + 1).bit_length() - 1
    n_first = n_chunks_max - (2**t - 1)
    if n_first > 0:
        has_first = n_full >= 2**t
        out = evaluate_if(has_first, chunk_size, n_first, out, 0)
        first = jnp.where(has_first, n_first, 0)
    else:
        first = 0
    n_other = n_full - first
    for k in reversed(range(t)):
        start = (first + ((n_other >> (k + 1)) << (k + 1))) * chunk_size
        out = evaluate_if(((n_other >> k) & 1) == 1, chunk_size, 1 << k, out, start)

    # The rows after the full chunks, in chunks of the powers of two smaller
    # than chunk_size, down to min_chunk_size, for every bit of their number.
    n_tail = n_selected - n_full * chunk_size
    for k in reversed(range((chunk_size - 1).bit_length())):
        size = 1 << k
        if size < min_chunk_size:
            break
        start = n_full * chunk_size + ((n_tail >> (k + 1)) << (k + 1))
        out = evaluate_if(((n_tail >> k) & 1) == 1, size, 1, out, start)

    # The last rows, fewer than min_chunk_size, in a padded chunk.
    if min_chunk_size > 1:
        start = n_full * chunk_size + (n_tail // min_chunk_size) * min_chunk_size
        out = evaluate_if(n_tail % min_chunk_size > 0, min_chunk_size, 1, out, start)
    return out


@custom_vmap
def _max_over_batch(n):
    # The identity, except under vmap: the largest value of the batch, the
    # same for all of its elements.
    return n


@_max_over_batch.def_vmap
def _max_over_batch_vmap_rule(axis_size, in_batched, n):
    return jnp.max(n, axis=0), False


def _apply_packed_jvp(impl, primals, tangents):
    # The native derivative of the implementation, with respect to the
    # arguments with a nonzero tangent only: the others are closed over, so
    # that f is not differentiated with respect to them. The custom vmap rule
    # is not used: reverse mode cannot transpose it.
    leaves, tree = jax.tree.flatten(primals)
    tangents = tree.flatten_up_to(tangents)
    nonzero = [type(t) is not SymbolicZero for t in tangents]

    def impl_nonzero(*nonzero_leaves):
        it = iter(nonzero_leaves)
        args = [next(it) if nz else p for p, nz in zip(leaves, nonzero)]
        return impl(*jax.tree.unflatten(tree, args))

    return jax.jvp(
        impl_nonzero,
        [p for p, nz in zip(leaves, nonzero) if nz],
        [t for t, nz in zip(tangents, nonzero) if nz],
    )


def _apply_packed_vmap_rule(
    impl, f, chunk_size, min_chunk_size, axis_size, in_batched, pars, x, where, fill
):
    pars_batched, x_batched, where_batched, fill_batched = in_batched
    rows_batched = where_batched or any(jax.tree.leaves(x_batched))

    if rows_batched and not any(jax.tree.leaves(pars_batched)):
        # Fold the batch into the rows, pack all of them at once, and unfold.
        shape = where.shape[1:] if where_batched else where.shape
        n = shape[0]

        def fold(a, batched):
            if not batched:
                a = jnp.broadcast_to(a, (axis_size,) + a.shape)
            return a.reshape((axis_size * n,) + a.shape[2:])

        x = jax.tree.map(fold, x, x_batched)
        where = fold(where, where_batched)

        out_struct = _row_struct(f, pars, x)

        def fold_fill(v, batched, s):
            full = shape + s.shape[1:]
            if batched:
                v = v.reshape(
                    v.shape[:1] + (1,) * (len(full) - v.ndim + 1) + v.shape[1:]
                )
                v = jnp.broadcast_to(v, (axis_size,) + full)
            else:
                v = jnp.broadcast_to(v, (axis_size,) + full)
            return v.reshape((axis_size * n,) + full[1:])

        fill = jax.tree.map(fold_fill, fill, fill_batched, out_struct)
        # Through the public function, so that a nested vmap is folded as well.
        out = apply_packed(
            f,
            pars,
            x,
            where,
            fill,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
        )
        out = jax.tree.map(
            lambda o: o.reshape((axis_size,) + shape + o.shape[len(shape) :]), out
        )
    elif rows_batched:
        # Both the rows and pars are batched: the elements of the batch are
        # evaluated one after the other, each with its own packing.
        leaves, tree = jax.tree.flatten((pars, x, where, fill))
        is_batched = tree.flatten_up_to(tuple(in_batched))

        def evaluate(batched_leaves):
            it = iter(batched_leaves)
            args = [next(it) if b else a for a, b in zip(leaves, is_batched)]
            return impl(*jax.tree.unflatten(tree, args))

        out = jax.lax.map(evaluate, [a for a, b in zip(leaves, is_batched) if b])
    else:
        # The packing only depends on where: with the rows unbatched, the
        # batch of pars (or fill) is evaluated in the same chunks.
        in_axes = jax.tree.map(lambda b: 0 if b else None, tuple(in_batched))
        out = jax.vmap(impl, in_axes=in_axes)(pars, x, where, fill)
    return out, jax.tree.map(lambda _: True, out)
