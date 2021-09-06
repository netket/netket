import jax
import jax.numpy as jnp

from jax import linear_util as lu
from jax.api_util import argnums_partial

# from functools import wraps


def scan_accum(f, x, op):
    res_init = jax.tree_map(
        lambda x: jnp.zeros(x.shape), jax.eval_shape(f, jax.tree_map(lambda x: x[0], x))
    )

    def f_(carry, x):
        return op(carry, f(x)), None

    res, _ = jax.lax.scan(
        f_, res_init, x, unroll=1
    )  # unroll=1 to make sure it uses the loop impl
    return res


def scan_append(f, x):
    def f_(carry, x):
        return None, f(x)

    _, res = jax.lax.scan(
        f_, None, x, unroll=1
    )  # unroll=1 to make sure it uses the loop impl
    return res


# TODO in_axes a la vmap?
def scanmap(fun, scan_fun, argnums=0):
    # @wraps(f)
    def f_(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args)
        return scan_fun(lambda x: f_partial.call_wrapped(*x), dyn_args)

    return f_
