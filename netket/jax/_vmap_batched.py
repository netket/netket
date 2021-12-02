from typing import Callable, Optional

import jax
import jax.numpy as jnp

from ._batch_utils import _batch, _unbatch
from ._scanmap import scanmap, scan_append


def _batch_vmapped_function(vmapped_fun, batch_size):
    """takes a vmapped function and computes it in batches"""

    # TODO axes

    def _fun(x):

        # TODO support pytrees
        n_elements = x.shape[0]

        n_batches, n_rest = divmod(n_elements, batch_size)

        if n_batches == 0 or batch_size >= n_elements:
            y = vmapped_fun(x)
        else:
            # split inputs
            x_batches = _batch(x[: n_elements - n_rest, ...], batch_size)
            x_rest = x[n_elements - n_rest :, ...]

            y_batches = _unbatch(scanmap(vmapped_fun, scan_append)(x_batches))

            if n_rest == 0:
                y = y_batches
            else:
                y_rest = vmapped_fun(x_rest)
                # TODO avoid this copy?
                # could preallocate + fori loop & index update
                y = jnp.concatenate((y_batches, y_rest))

        return y

    return _fun


def vmap_batched(f: Callable, in_axes=0, *, batch_size: Optional[int]):
    """
    Behaves like jax.vmap but uses scan to batch the computations in smaller batches.
    """

    if in_axes != 0:
        raise ValueError("Only in_axes=0 is supported")

    vmapped_fun = jax.vmap(f, in_axes=0)

    if batch_size is None:
        return vmapped_fun
    else:
        return _batch_vmapped_function(vmapped_fun, batch_size)
