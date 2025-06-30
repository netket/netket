from typing import Union, Iterable

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
from jax._src.lax.control_flow.loops import _batch_and_remainder


def split_tuple(
    data: tuple, indices: Union[int, Iterable[int]]
) -> tuple[tuple, tuple, tuple[int, ...]]:
    """
    Splits `data` into two tuples based on `indices`.

    Args:
        data: A tuple of values.
        indices: An int or iterable of ints indicating which positions to select.

    Returns:
        selected:   A tuple of data[i] for each i in indices (in the order given).
        remainder:  A tuple of the other elements of data (in their original order).
        indices:    A tuple of the indices used (normalized).
    """
    # Normalize indices to a tuple of ints
    if isinstance(indices, int):
        idxs = (indices,)
    else:
        idxs = tuple(indices)

    # Build the selected and remainder tuples
    selected = tuple(data[i] for i in idxs)
    remainder = tuple(val for pos, val in enumerate(data) if pos not in idxs)
    return selected, remainder, idxs


def recompose_tuple(
    selected: tuple,
    remainder: tuple,
    indices: Iterable[int],
) -> tuple:
    """
    Reconstructs the original tuple from `selected`, `remainder`, and `indices`.

    Args:
        selected:  Tuple of items that belong at the given indices.
        remainder: Tuple of the other items, in their original order.
        indices:   Iterable of positions at which the selected items go.

    Returns:
        A tuple of length len(selected) + len(remainder), with
        `selected` items re-inserted at `indices` and `remainder`
        filling the other slots.
    """
    idxs = tuple(indices)
    total_len = len(selected) + len(remainder)
    result = [None] * total_len

    # Place selected items
    for sel_idx, orig_pos in enumerate(idxs):
        result[orig_pos] = selected[sel_idx]

    # Fill in the rest from remainder, in order
    rem_iter = iter(remainder)
    for i in range(total_len):
        if result[i] is None:
            result[i] = next(rem_iter)

    return tuple(result)


def reduce(
    fun,
    xs,
    *,
    init_fun=jnp.zeros_like,
    reduce_fun=lambda a, b: a + b,
    stack_outnums: int | tuple[int] = (),
    batch_size=None,
):
    """
    Chunk-and-fold reduction:

    reduce(fun, xs, batch_size=None) == fun(xs)

    reduce(fun, xs, batch_size=B) ==
       fold(reduce_fun, init, [ fun(chunk) for chunk in chunks_of_size_B(xs) ])

    Args:
      fun:        callable taking a pytree `xs_chunk` and returning some pytree output.
      xs:         pytree of arrays to be reduced along leading axis.
      init_fun:   fn(abstract_out) -> initial accumulator (default zeros_like).
      reduce_fun: binary fn(acc, out) -> new_acc (default add).
      batch_size: if None, just do fun(xs); else slice into batches of this size.
      stack_outnums: if not empty, stack the outputs of fun at these indices instead of reducing them.
        (This is useful when using fun to compute a vjp, where the first output is the forward pass
        to be stacked and the second output is the gradient to be reduced.)

    Returns:
      A single pytree matching `fun`'s output, the reduction of all chunks.
    """
    # If no batching, just do it in one go.
    if batch_size is None:
        return fun(xs)

    # This function has two outputs: the stacked output and the reduced output
    def fun_with_stack(xs):
        out_stack, out_reduce, _ = split_tuple(fun(xs), stack_outnums)
        return out_stack, out_reduce

    # Infer the shape/dtype of fun(xs) for initializing the accumulator.
    result_stack_aval, result_reduce_aval = jax.eval_shape(fun_with_stack, xs)
    reduce_output_init = tree_map(init_fun, result_reduce_aval)

    # Split xs into full batches + optional remainder
    scan_xs, rem_xs = _batch_and_remainder(xs, batch_size)

    # Scan body: run fun(chunk) and reduce the corresponding output.
    def _body(acc, x_chunk):
        out_stack, out_reduce = fun_with_stack(x_chunk)
        acc2 = tree_map(reduce_fun, acc, out_reduce)
        return acc2, out_stack

    # Fold over all full batches
    acc_batches, stacked_out = lax.scan(_body, reduce_output_init, scan_xs)

    # Finally fold in the remainder, if any
    if rem_xs is not None:
        acc_final, _rem_out = _body(acc_batches, rem_xs)
    else:
        acc_final = acc_batches
        _rem_out = jax.tree.map(lambda x: None, result_stack_aval)

    def _stack_stuff(out, rem):
        out = out.reshape(-1, *out.shape[2:])
        if rem is not None:
            out = jnp.concatenate([out, rem], axis=0)
        return out

    stacked_out = jax.tree.map(_stack_stuff, stacked_out, _rem_out)

    outputs = recompose_tuple(stacked_out, acc_final, stack_outnums)
    return outputs


# # # — assume your `reduce` from above is already in scope —
# stack = False


# # 1) A simple linear model: f(W, xs) = xs @ W  → outputs shape (batch,)
# def model(pars, xs):
#     W, b = pars
#     return xs @ W + b


# # 2) Wrap a chunk‐wise VJP into a function of just the chunked pytree (xs_chunk, v_chunk):
# def make_chunk_vjp(W, *, stack=False):
#     def chunk_grad(batch):
#         xs_chunk, v_chunk = batch
#         # vjp on the linear model w.r.t. W
#         val, vjp_fun = jax.vjp(lambda W_: model(W_, xs_chunk), W)
#         (gW,) = vjp_fun(v_chunk)  # gradient wrt W only
#         if stack:
#             return val, gW
#         else:
#             return gW

#     return chunk_grad


# # from jax.sharding import PartitionSpec as P, AxisType, reshard

# # if jax.sharding.get_abstract_mesh().empty:
# #     jax.config.update("jax_num_cpu_devices", 4)
# #     mesh = jax.make_mesh((4,), ('S'), axis_types=(AxisType.Explicit,))
# #     jax.sharding.set_mesh(mesh)


# # 3) Create some fake data
# key = jax.random.PRNGKey(0)
# M, N = 5, 100
# W = jax.random.normal(key, (M,))  # our “parameters”
# b = jax.random.normal(key, ())  # our “parameters”
# pars = (W, b)  # model parameters
# xs = jax.random.normal(key, (N, M))  # N samples of dimension M
# v = jax.random.normal(key, (N,))  # cotangent vector of length N

# # 4) Full‐dataset VJP (no batching)
# _, vjp_fun_full = jax.vjp(lambda W_: model(W_, xs), pars)
# (grad_full,) = vjp_fun_full(v)

# # 5) Chunk‐and‐reduce VJP
# data = (xs, v)
# chunk_fun = make_chunk_vjp(pars, stack=False)
# grad_batched = reduce(chunk_fun, data, batch_size=32, stack_first_output=False)

# # 6) Compare
# print("full‐VJP grad:", grad_full)
# print("batched‐reduce grad:", grad_batched)

# data = (xs, v)
# chunk_fun = make_chunk_vjp(pars, stack=True)
# fwd, grad_batched = reduce(chunk_fun, data, batch_size=4, stack_first_output=True)

# # 6) Compare
# print("full‐VJP grad:", grad_full)
# print("batched‐reduce grad:", grad_batched)

# # xss = reshard(xs, P('S'))
# # vs = reshard(v, P('S'))
# # datas = (xss, vs)
# # fwd, grad_batched = reduce(chunk_fun, datas, batch_size=4, stack_first_output=True)
