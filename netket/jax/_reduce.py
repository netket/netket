import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
from jax._src.lax.control_flow.loops import _batch_and_remainder


def reduce(
    fun,
    xs,
    *,
    init_fun=jnp.zeros_like,
    reduce_fun=lambda a, b: a + b,
    stack_first_output: bool = False,
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

    Returns:
      A single pytree matching `fun`'s output, the reduction of all chunks.
    """
    # If no batching, just do it in one go.
    if batch_size is None:
        return fun(xs)

    # Infer the shape/dtype of fun(xs) for initializing the accumulator.
    result_aval = jax.eval_shape(fun, xs)
    if stack_first_output:
        result_aval = result_aval[1:]
    acc0 = tree_map(init_fun, result_aval)

    # Split xs into full batches + optional remainder
    scan_xs, rem_xs = _batch_and_remainder(xs, batch_size)

    # Scan body: run fun(chunk) and fold into acc
    def _body(acc, x_chunk):
        out = fun(x_chunk)
        if stack_first_output:
            out1, *out = out
            out = tuple(out)
        else:
            out1 = ()
        acc2 = tree_map(reduce_fun, acc, out)
        # we don't need to collect per‐batch outputs, so return an empty pytree
        return acc2, out1

    # Fold over all full batches
    acc_batches, stacked_out = lax.scan(_body, acc0, scan_xs)

    # Finally fold in the remainder, if any
    if rem_xs is not None:
        acc_final, _rem_out = _body(acc_batches, rem_xs)
    else:
        acc_final = acc_batches
        _rem_out = None

    if stack_first_output:
        stacked_out = stacked_out.reshape(
            -1, *stacked_out.shape[2:]
        )  # stack the first output
        if _rem_out is not None:
            stacked_out = jnp.concatenate([stacked_out, _rem_out], axis=0)

        # in the body of the scan we unpacked a tuple, but if it has length 2 this
        # nests stuff into another tuple, so we need to flatten it
        if len(result_aval) == 2:
            return stacked_out, acc_final[0]
        return stacked_out, acc_final
    else:
        return acc_final


# # # — assume your `reduce` from above is already in scope —
# stack = False


# # 1) A simple linear model: f(W, xs) = xs @ W  → outputs shape (batch,)
# def model(pars, xs):
#     W, b = pars
#     return xs @ W + b


# # 2) Wrap a chunk‐wise VJP into a function of just the chunked pytree (xs_chunk, v_chunk):
# def make_chunk_vjp(W, v, *, stack=False):
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
# chunk_fun = make_chunk_vjp(pars, v, stack=False)
# grad_batched = reduce(chunk_fun, data, batch_size=32, stack_first_output=False)

# # 6) Compare
# print("full‐VJP grad:", grad_full)
# print("batched‐reduce grad:", grad_batched)

# data = (xs, v)
# chunk_fun = make_chunk_vjp(pars, v, stack=True)
# fwd, grad_batched = reduce(chunk_fun, data, batch_size=32, stack_first_output=True)

# # 6) Compare
# print("full‐VJP grad:", grad_full)
# print("batched‐reduce grad:", grad_batched)
