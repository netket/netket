import jax
import jax.numpy as jnp

from jax import linear_util as lu
from jax.api_util import argnums_partial

from functools import partial

# from functools import wraps


_tree_add = partial(jax.tree_multimap, jax.lax.add)
_tree_zeros_like = partial(jax.tree_map, lambda x: jnp.zeros(x.shape))


# TODO put it somewhere
def _multimap(f, *args):
    try:
        return tuple(map(lambda a: f(*a), zip(*args)))
    except TypeError:
        return f(*args)


def scan_append_accum(f, x, append_cond, op=_tree_add):
    """Evaluate f element by element in x while appending and/or accumulating the results

    Args:
        f: a function that takes elements of the leading dimension of x
        x: a pytree where each leaf array has the same leading dimension
        append_cond: a bool (if f returns just one result) or a tuple of bools (if f returns multiple values)
            which indicates whether the individual result should be appended or accumulated
        op: a function to accumulate the specified results. Defaults to a sum.
    Returns:
        returns the (tuple of) results corresponding to the output of f
        where each result is given by:
        - a (pytree of) array(s) with leading dimension same as x, containing the evaluation of f at each element in x
        - a (pytree of) array with the same shape as the corresponding output of f, containg the reduction over op of f evaluated at each x"""
    # TODO: custom initialization (not just 0.) (or avoid it?)
    # TODO: different op for each result

    x0 = jax.tree_map(lambda x: x[0], x)

    # special code path if there is only one element
    # to avoid having to rely on xla/llvm to optimize the overhead away
    if jax.tree_leaves(x)[0].shape[0] == 1:
        return jax.tree_map(partial(jnp.expand_dims, axis=0), f(x0))

    # the original idea was to use pytrees, however for now just operate on the return value tuple
    _get_append_part = partial(_multimap, lambda c, x: x if c else None, append_cond)
    _get_op_part = partial(_multimap, lambda c, x: x if not c else None, append_cond)
    _tree_select = partial(_multimap, lambda c, t1, t2: t1 if c else t2, append_cond)

    carry_init = True, _get_op_part(_tree_zeros_like(jax.eval_shape(f, x0)))

    def f_(carry, x):
        is_first, y_carry = carry
        y = f(x)
        y_op = _get_op_part(y)
        y_append = _get_append_part(y)
        # select here to avoid the user having to specify the zero element for op
        y_accum = jax.tree_multimap(
            partial(jax.lax.select, is_first), y_op, op(y_carry, y_op)
        )
        return (False, y_accum), y_append

    (_, res_op), res_append = jax.lax.scan(f_, carry_init, x, unroll=1)
    # reconstruct the result from the accumulated and appended parts in the two trees
    return _tree_select(res_append, res_op)


scan_append = partial(scan_append_accum, append_cond=True)
scan_accum = partial(scan_append_accum, append_cond=False)


# TODO in_axes a la vmap?
def scanmap(fun, scan_fun, argnums=0):
    # @wraps(f)
    def f_(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args)
        return scan_fun(lambda x: f_partial.call_wrapped(*x), dyn_args)

    return f_
