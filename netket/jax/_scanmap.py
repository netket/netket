import jax
import jax.numpy as jnp

from jax import linear_util as lu
from jax.api_util import argnums_partial as _argnums_partial

from functools import partial, wraps

from netket.utils import module_version

_tree_add = partial(jax.tree_map, jax.lax.add)
_tree_zeros_like = partial(jax.tree_map, lambda x: jnp.zeros(x.shape, dtype=x.dtype))


# TODO put it somewhere
def _multimap(f, *args):
    try:
        return tuple(map(lambda a: f(*a), zip(*args)))
    except TypeError:
        return f(*args)


# TODO: When minimum jax is v0.2.22, remove this function and import directly
# argnums_partial.
# This works around onled argnums_partial implementations that did not have
# require_static_args_hashable kwarg (which was implicitly False).
if module_version(jax) >= (0, 2, 22):
    argnums_partial = _argnums_partial
else:

    @wraps(_argnums_partial)
    def argnums_partial(*args, require_static_args_hashable=True, **kwargs):
        return _argnums_partial(*args, **kwargs)


def scan_append_reduce(f, x, append_cond, op=_tree_add):
    """Evaluate f element by element in x while appending and/or reducing the results

    Args:
        f: a function that takes elements of the leading dimension of x
        x: a pytree where each leaf array has the same leading dimension
        append_cond: a bool (if f returns just one result) or a tuple of bools (if f returns multiple values)
            which indicates whether the individual result should be appended or reduced
        op: a function to (pairwise) reduce the specified results. Defaults to a sum.
    Returns:
        returns the (tuple of) results corresponding to the output of f
        where each result is given by:
        if append_cond is True:
            a (pytree of) array(s) with leading dimension same as x,
            containing the evaluation of f at each element in x
        else (append_cond is False):
            a (pytree of) array(s) with the same shape as the corresponding output of f,
            containg the reduction over op of f evaluated at each x


    Example:

        import jax.numpy as jnp
        from netket.jax import scan_append_reduce

        def f(x):
             y = jnp.sin(x)
             return y, y, y**2

        N = 100
        x = jnp.linspace(0.,jnp.pi,N)

        y, s, s2 = scan_append_reduce(f, x, (True, False, False))
        mean = s/N
        var = s2/N - mean**2
    """
    # TODO: different op for each result

    x0 = jax.tree_map(lambda x: x[0], x)

    # special code path if there is only one element
    # to avoid having to rely on xla/llvm to optimize the overhead away
    if jax.tree_leaves(x)[0].shape[0] == 1:
        return _multimap(
            lambda c, x: jnp.expand_dims(x, 0) if c else x, append_cond, f(x0)
        )

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
        y_reduce = jax.tree_map(
            partial(jax.lax.select, is_first), y_op, op(y_carry, y_op)
        )
        return (False, y_reduce), y_append

    (_, res_op), res_append = jax.lax.scan(f_, carry_init, x, unroll=1)
    # reconstruct the result from the reduced and appended parts in the two trees
    return _tree_select(res_append, res_op)


scan_append = partial(scan_append_reduce, append_cond=True)
scan_reduce = partial(scan_append_reduce, append_cond=False)


# TODO in_axes a la vmap?
def scanmap(fun, scan_fun, argnums=0):
    """
    A helper function to wrap f with a scan_fun

    Example:
        import jax.numpy as jnp
        from functools import partial

        from netket.jax import scanmap, scan_append_reduce

        scan_fun = partial(scan_append_reduce, append_cond=(True, False, False))

        @partial(scanmap, scan_fun=scan_fun, argnums=1)
        def f(c, x):
             y = jnp.sin(x) + c
             return y, y, y**2

        N = 100
        x = jnp.linspace(0.,jnp.pi,N)
        c = 1.


        y, s, s2 = f(c, x)
        mean = s/N
        var = s2/N - mean**2
    """

    def f_(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False
        )
        return scan_fun(lambda x: f_partial.call_wrapped(*x), dyn_args)

    return f_
