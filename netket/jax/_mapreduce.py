import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
from typing import Callable, Any

# ——— batched splitting (copy‐paste from lax.map) ———

# def _scan_leaf(leaf, batch_elems, num_batches, batch_size):
#     def f(l):
#         return l[:batch_elems].reshape(num_batches, batch_size, *l.shape[1:])
#     aval = jax.core.typeof(leaf)
#     return f(leaf)


# def _remainder_leaf(leaf, batch_elems):
#     def f(l):
#         return l[batch_elems:]

#     return f(leaf)


# def _batch_and_remainder(x, batch_size: int):
#     leaves, treedef = tree_flatten(x)
#     if not leaves:
#         return x, None
#     n, r = divmod(leaves[0].shape[0], batch_size)
#     batch_elems = n * batch_size
#     if r:
#         scan_leaves, rem_leaves = zip(
#             *[
#                 (
#                     _scan_leaf(leaf, batch_elems, n, batch_size),
#                     _remainder_leaf(leaf, batch_elems),
#                 )
#                 for leaf in leaves
#             ]
#         )
#         return treedef.unflatten(scan_leaves), treedef.unflatten(rem_leaves)
#     else:
#         scan_leaves = [_scan_leaf(leaf, batch_elems, n, batch_size) for leaf in leaves]
#         return treedef.unflatten(scan_leaves), None


# ——— the mapreduce itself ———
from jax._src.lax.control_flow.loops import _batch_and_remainder


def mapreduce(
    f: Callable[[Any], Any],
    xs: Any,
    *,
    init_fn: Callable[[Any], Any] = jnp.zeros_like,
    reduce_fn: Callable[[Any, Any], Any] = lambda a, b: a + b,
    batch_size: int | None = None,
) -> Any:
    """
    Map `f` over the leading axis of `xs` and reduce all results into one accumulator.

    Args:
      f:          function taking one slice of `xs` and returning a pytree of arrays.
      xs:         pytree of arrays, all with the same leading dimension N.
      init_fn:    fn(abstract‐value) → initial accumulator leaf (default zeros).
      reduce_fn:  binary fn(acc_leaf, out_leaf) → new_acc_leaf (default sum).
      batch_size: if provided, splits N elements into minibatches of this size.

    Returns:
      acc_final: pytree of the same structure as `f`’s output, where each leaf
                 is the reduction (via `reduce_fn`) of all f(x) along the leading axis.
    """
    # 1) infer output shape “abstractly” from a single example
    example = tree_map(lambda arr: arr[0], xs)
    result_aval = jax.eval_shape(f, example)

    # 2) init accumulator
    acc0 = tree_map(init_fn, result_aval)

    # helper: fold a batch of ys into acc via scan
    def _reduce_batch(acc, y_batch):
        def step(c, y):
            return reduce_fn(c, y), None

        acc_out, _ = lax.scan(step, acc, y_batch)
        return acc_out

    if batch_size is None:
        # scan one by one
        def body(acc, x):
            y = f(x)
            new_acc = tree_map(reduce_fn, acc, y)
            return new_acc, None  # drop y

        acc_final, _ = lax.scan(body, acc0, xs)

    else:
        # split into full batches + optional remainder
        scan_xs, rem_xs = _batch_and_remainder(xs, batch_size)

        # scan over full batches
        def batch_body(acc, xb):
            yb = jax.vmap(f)(xb)  # (batch_size, …)
            acc2 = _reduce_batch(acc, yb)  # fold batch
            return acc2, None

        acc_batches, _ = lax.scan(batch_body, acc0, scan_xs)

        # handle remainder
        if rem_xs is not None:
            y_rem = jax.vmap(f)(rem_xs)
            acc_final = _reduce_batch(acc_batches, y_rem)
        else:
            acc_final = acc_batches

    return acc_final
