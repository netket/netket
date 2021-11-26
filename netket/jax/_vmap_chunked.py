from typing import Callable, Optional

import jax
import jax.numpy as jnp

from ._chunk_utils import _chunk, _unchunk
from ._scanmap import scanmap, scan_append


def _chunk_vmapped_function(vmapped_fun, chunk_size):
    """takes a vmapped function and computes it in chunks"""

    # TODO axes

    def _fun(x):

        # TODO support pytrees
        n_elements = x.shape[0]

        n_chunks, n_rest = divmod(n_elements, chunk_size)

        if n_chunks == 0 or chunk_size >= n_elements:
            y = vmapped_fun(x)
        else:
            # split inputs
            x_chunks = _chunk(x[: n_elements - n_rest, ...], chunk_size)
            x_rest = x[n_elements - n_rest :, ...]

            y_chunks = _unchunk(scanmap(vmapped_fun, scan_append)(x_chunks))

            if n_rest == 0:
                y = y_chunks
            else:
                y_rest = vmapped_fun(x_rest)
                # TODO avoid this copy?
                # could preallocate + fori loop & index update
                y = jnp.concatenate((y_chunks, y_rest))

        return y

    return _fun


def vmap_chunked(f: Callable, in_axes=0, *, chunk_size: Optional[int]):
    """
    Behaves like jax.vmap but uses scan to chunk the computations in smaller chunks.
    """

    if in_axes != 0:
        raise ValueError("Only in_axes=0 is supported")

    vmapped_fun = jax.vmap(f, in_axes=0)

    if chunk_size is None:
        return vmapped_fun
    else:
        return _chunk_vmapped_function(vmapped_fun, chunk_size)
