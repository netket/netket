from typing import Callable, Optional

import jax
import jax.numpy as jnp

from ._chunk_utils import _chunk, _unchunk
from ._scanmap import scanmap, scan_append


def _chunk_vmapped_function(vmapped_fun, chunk_size, argnums=0):
    """takes a vmapped function and computes it in chunks"""

    if chunk_size is None:
        return vmapped_fun

    if isinstance(argnums, int):
        argnums = (argnums,)

    def _fun(*args, **kwargs):

        n_elements = jax.tree_util.tree_leaves(args[argnums[0]])[0].shape[0]
        n_chunks, n_rest = divmod(n_elements, chunk_size)

        if n_chunks == 0 or chunk_size >= n_elements:
            y = vmapped_fun(*args, **kwargs)
        else:
            # split inputs
            def _get_chunks(x):
                x_chunks = jax.tree_map(lambda x_: x_[: n_elements - n_rest, ...], x)
                x_chunks = _chunk(x_chunks, chunk_size)
                return x_chunks

            def _get_rest(x):
                x_rest = jax.tree_map(lambda x_: x_[n_elements - n_rest :, ...], x)
                return x_rest

            args_chunks = [
                _get_chunks(a) if i in argnums else a for i, a in enumerate(args)
            ]
            args_rest = [
                _get_rest(a) if i in argnums else a for i, a in enumerate(args)
            ]

            y_chunks = _unchunk(
                scanmap(vmapped_fun, scan_append, argnums)(*args_chunks, **kwargs)
            )

            if n_rest == 0:
                y = y_chunks
            else:
                y_rest = vmapped_fun(*args_rest, **kwargs)
                y = jax.tree_map(
                    lambda y1, y2: jnp.concatenate((y1, y2)), y_chunks, y_rest
                )
        return y

    return _fun


def _parse_in_axes(in_axes):
    
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    if not set(in_axes).issubset((0, None)):
        raise NotImplementedError("Only in_axes 0/None are currently supported")

    argnums = tuple(
        map(lambda ix: ix[0], filter(lambda ix: ix[1] is not None, enumerate(in_axes)))
    )
    return in_axes, argnums


def apply_chunked(f: Callable, in_axes=0, *, chunk_size: Optional[int]):
    """
    Takes a (explicitly or implicitly) vmapped function and uses scan to
    do the computations in smaller chunks.
    """

    _, argnums = _parse_in_axes(in_axes)
    return _chunk_vmapped_function(f, chunk_size, argnums)


def vmap_chunked(f: Callable, in_axes=0, *, chunk_size: Optional[int]):
    """
    Behaves like jax.vmap but uses scan to chunk the computations in smaller chunks.
    """

    in_axes, argnums = _parse_in_axes(in_axes)
    vmapped_fun = jax.vmap(f, in_axes=in_axes)
    return _chunk_vmapped_function(vmapped_fun, chunk_size, argnums)
