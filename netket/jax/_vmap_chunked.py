from collections.abc import Callable

import jax

from ._chunk_utils import _chunk, _unchunk
from ._scanmap import scanmap, scan_append

from netket.utils import HashablePartial
from netket.utils import config
from netket.jax.sharding import sharding_decorator


def _eval_fun_in_chunks(vmapped_fun, chunk_size, argnums, *args, **kwargs):
    # split inputs
    args_chunks, args_rest = zip(
        *[
            _chunk(a, chunk_size=chunk_size) if i in argnums else (a, a)
            for i, a in enumerate(args)
        ]
    )

    n_chunks = jax.tree_util.tree_leaves(args_chunks[argnums[0]])[0].shape[0]
    n_rest = jax.tree_util.tree_leaves(args_rest[argnums[0]])[0].shape[0]

    if n_chunks > 0:
        y_chunks = scanmap(vmapped_fun, scan_append, argnums)(*args_chunks, **kwargs)
    if n_rest > 0:
        y_rest = vmapped_fun(*args_rest, **kwargs)

    if n_chunks > 0 and n_rest > 0:
        return _unchunk(y_chunks, y_rest)
    elif n_chunks > 0:
        return _unchunk(y_chunks)
    elif n_rest > 0:
        return y_rest
    else:
        return vmapped_fun(*args, **kwargs)


def _eval_fun_in_chunks_sharding(vmapped_fun, chunk_size, argnums, *args, **kwargs):
    # Equivalent to `_eval_fun_in_chunks` above but preserves sharding,
    # by computing the vmapped_fun in chunks on every shard (which sits on a separate device)
    sharded_args_tree = tuple(i in argnums for i, a in enumerate(args))
    f = HashablePartial(_eval_fun_in_chunks, vmapped_fun, chunk_size, argnums, **kwargs)

    # if the vmapped_fun e.g. does a vjp we need to make the params pvary here
    # to avoid it emitting an unwanted psum
    pvary_argnums = tuple(set(range(len(args))).difference(argnums))
    pvary_args_tree = tuple(i in pvary_argnums for i in range(len(args)))

    return sharding_decorator(f, sharded_args_tree, pvary_args_tree=pvary_args_tree)(
        *args
    )


def _chunk_vmapped_function(
    vmapped_fun: Callable,
    chunk_size: int | None,
    argnums: int | tuple[int, ...] = 0,
    axis_0_is_sharded: bool = False,
) -> Callable:
    """takes a vmapped function and computes it in chunks"""

    if chunk_size is None:
        # TODO: Here we are not applyign sharding_decorator, while below we are
        # Maybe we should always apply it for consistency?
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
    chunk_size: int | None,
    axis_0_is_sharded: bool = None,  # type: ignore[attr-defined]
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
    if axis_0_is_sharded is None:
        axis_0_is_sharded = config.netket_experimental_sharding

    _, argnums = _parse_in_axes(in_axes)
    return _chunk_vmapped_function(f, chunk_size, argnums, axis_0_is_sharded)


def vmap_chunked(
    f: Callable,
    in_axes=0,
    *,
    chunk_size: int | None,
    axis_0_is_sharded: bool = None,  # type: ignore[attr-defined]
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
    if axis_0_is_sharded is None:
        axis_0_is_sharded = config.netket_experimental_sharding

    in_axes, argnums = _parse_in_axes(in_axes)
    vmapped_fun = jax.vmap(f, in_axes=in_axes)
    return _chunk_vmapped_function(vmapped_fun, chunk_size, argnums, axis_0_is_sharded)
