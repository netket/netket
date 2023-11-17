from typing import Callable, Optional

import jax
import jax.numpy as jnp

from ._chunk_utils import _chunk, _unchunk
from ._scanmap import scanmap, scan_append

from netket.utils import HashablePartial
from netket.utils import config
from netket.jax.sharding import sharding_decorator


def _eval_fun_in_chunks(vmapped_fun, chunk_size, argnums, *args, **kwargs):
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
        args_rest = [_get_rest(a) if i in argnums else a for i, a in enumerate(args)]

        y_chunks = _unchunk(
            scanmap(vmapped_fun, scan_append, argnums)(*args_chunks, **kwargs)
        )

        if n_rest == 0:
            y = y_chunks
        else:
            y_rest = vmapped_fun(*args_rest, **kwargs)
            y = jax.tree_map(lambda y1, y2: jnp.concatenate((y1, y2)), y_chunks, y_rest)
    return y


def _eval_fun_in_chunks_sharding(vmapped_fun, chunk_size, argnums, *args, **kwargs):
    # Equivalent to `_eval_fun_in_chunks` above but preserves sharding,
    # by computing the vmapped_fun in chunks on every shard (which sits on a separate device)
    sharded_args_tree = tuple(i in argnums for i, a in enumerate(args))
    f = HashablePartial(_eval_fun_in_chunks, vmapped_fun, chunk_size, argnums, **kwargs)
    return sharding_decorator(f, sharded_args_tree)(*args)


def _chunk_vmapped_function(
    vmapped_fun: Callable,
    chunk_size: Optional[int],
    argnums=0,
    axis_0_is_sharded=False,
) -> Callable:
    """takes a vmapped function and computes it in chunks"""

    if chunk_size is None:
        return vmapped_fun

    if isinstance(argnums, int):
        argnums = (argnums,)
    if axis_0_is_sharded:
        return HashablePartial(
            _eval_fun_in_chunks_sharding, vmapped_fun, chunk_size, argnums
        )
    else:
        return HashablePartial(_eval_fun_in_chunks, vmapped_fun, chunk_size, argnums)


def _parse_in_axes(in_axes):
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    if not set(in_axes).issubset((0, None)):
        raise NotImplementedError("Only in_axes 0/None are currently supported")

    argnums = tuple(
        map(lambda ix: ix[0], filter(lambda ix: ix[1] is not None, enumerate(in_axes)))
    )
    return in_axes, argnums


def apply_chunked(
    f: Callable,
    in_axes=0,
    *,
    chunk_size: Optional[int],
    axis_0_is_sharded=config.netket_experimental_sharding,
) -> Callable:
    """
    Takes an implicitly vmapped function over the axis 0 and uses scan to
    do the computations in smaller chunks over the 0-th axis of all input arguments.

    For this to work, the function `f` should be `vectorized` along the `in_axes`
    of the arguments. This means that the function `f` should respect the following
    condition:

    .. code-block:: python

        assert f(x) == jnp.concatenate([f(x_i) for x_i in x], axis=0)

    which is automatically satisfied if `f` is obtained by vmapping a function,
    such as:

    .. code-block:: python

        f = jax.vmap(f_orig)

    .. note::
        If netket_experimental_sharding is enabled, this function assumes that chunked in_axes are sharded by default.
        This can be overridden by specifying axis_0_is_sharded=False.

    Args:
        f: A function that satisfies the condition above
        in_axes: The axes that should be scanned along. Only supports `0` or `None`
        chunk_size: The maximum size of the chunks to be used. If it is `None`, chunking
            is disabled
        axis_0_is_sharded: specifies if axis 0 of the arrays scanned is sharded among multiple devices,
            The function is then computed in chunks of size chunk_size on every device.
            Defaults True if config.netket_experimental_sharding, oterhwise defaults to False.

    """
    _, argnums = _parse_in_axes(in_axes)
    return _chunk_vmapped_function(f, chunk_size, argnums, axis_0_is_sharded)


def vmap_chunked(
    f: Callable,
    in_axes=0,
    *,
    chunk_size: Optional[int],
    axis_0_is_sharded=config.netket_experimental_sharding,
) -> Callable:
    """
    Behaves like jax.vmap but uses scan to chunk the computations in smaller chunks.

    This function is essentially equivalent to:

    .. code-block:: python

        nk.jax.apply_chunked(jax.vmap(f, in_axes), in_axes, chunk_size)

    Some limitations to `in_axes` apply.

    .. note::
        If netket_experimental_sharding is enabled, this function assumes that chunked in_axes are sharded by default.
        This can be overridden by specifying axis_0_is_sharded=False.

    Args:
        f: The function to be vectorised.
        in_axes: The axes that should be scanned along. Only supports `0` or `None`
        chunk_size: The maximum size of the chunks to be used. If it is `None`, chunking
            is disabled
        axis_0_is_sharded: specifies if axis 0 of the arrays scanned is sharded among multiple devices,
            The function is then computed in chunks of size chunk_size on every device.
            Defaults True if config.netket_experimental_sharding, oterhwise defaults to False.

    Returns:
        A vectorised and chunked function
    """
    in_axes, argnums = _parse_in_axes(in_axes)
    vmapped_fun = jax.vmap(f, in_axes=in_axes)
    return _chunk_vmapped_function(vmapped_fun, chunk_size, argnums, axis_0_is_sharded)
