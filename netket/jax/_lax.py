from functools import partial
import jax


@partial(jax.jit, static_argnums=1)
def reduce_xor(x, axes):
    """
    equivalent of jax.lax.reduce_xor,
    with support for negative axes (counted from the back)
    """
    if isinstance(axes, int):
        axes = (axes,)
    axes = tuple(i if i >= 0 else x.ndim + i for i in axes)
    return jax.lax.reduce_xor_p.bind(x, axes=axes)
