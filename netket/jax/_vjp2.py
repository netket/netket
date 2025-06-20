from functools import partial, wraps
from typing import Any, Callable, Tuple, Union, Iterable

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P, AxisType
from jax.experimental.shard import reshard
from jax.experimental.shard_map import shard_map

from netket.utils.jax import HashablePartial

from ._reduce import reduce
from ._vjp import vjp as nkvjp


def split_tuple(
    data: Tuple, indices: Union[int, Iterable[int]]
) -> Tuple[Tuple, Tuple, Tuple[int, ...]]:
    """
    Splits `data` into two tuples based on `indices`.

    Args:
        data: A tuple of values.
        indices: An int or iterable of ints indicating which positions to select.

    Returns:
        selected:   A tuple of data[i] for each i in indices (in the order given).
        remainder:  A tuple of the other elements of data (in their original order).
        indices:    A tuple of the indices used (normalized).
    """
    # Normalize indices to a tuple of ints
    if isinstance(indices, int):
        idxs = (indices,)
    else:
        idxs = tuple(indices)

    # Build the selected and remainder tuples
    selected = tuple(data[i] for i in idxs)
    remainder = tuple(val for pos, val in enumerate(data) if pos not in idxs)
    return selected, remainder, idxs


def recompose_tuple(
    selected: Tuple,
    remainder: Tuple,
    indices: Iterable[int],
) -> Tuple:
    """
    Reconstructs the original tuple from `selected`, `remainder`, and `indices`.

    Args:
        selected:  Tuple of items that belong at the given indices.
        remainder: Tuple of the other items, in their original order.
        indices:   Iterable of positions at which the selected items go.

    Returns:
        A tuple of length len(selected) + len(remainder), with
        `selected` items re-inserted at `indices` and `remainder`
        filling the other slots.
    """
    idxs = tuple(indices)
    total_len = len(selected) + len(remainder)
    result = [None] * total_len

    # Place selected items
    for sel_idx, orig_pos in enumerate(idxs):
        result[orig_pos] = selected[sel_idx]

    # Fill in the rest from remainder, in order
    rem_iter = iter(remainder)
    for i in range(total_len):
        if result[i] is None:
            result[i] = next(rem_iter)

    return tuple(result)


# -


# ——— shard_map + lax.scan version ———
def pvarying_replicated(tree, axis_name):
    mesh = jax.sharding.get_abstract_mesh()
    n_devices = mesh.shape[axis_name]

    def _pvarying_replicated(x):
        xt = jax.typeof(x)
        if (
            isinstance(xt.sharding, jax.sharding.NamedSharding)
            and xt.sharding.spec[0] == axis_name
        ):
            return x
        x_shard = jnp.repeat(x[None, ...], n_devices, axis=0)  # shape (num_batches, M)
        x_shard = reshard(x_shard, P(axis_name, *(None for _ in range(x.ndim))))
        x_shard = x_shard.reshape(-1, *x.shape[1:])
        return x_shard

    return jax.tree.map(_pvarying_replicated, tree)


def stack_shards(tree, axis: int = 0):
    def _reshape(x):
        xt = jax.typeof(x)
        axis0_name = xt.sharding.mesh.axis_names[axis]
        n_devices = xt.sharding.mesh.shape[axis0_name]
        return x.reshape(n_devices, *x.shape[:axis], -1, *x.shape[axis + 1 :])

    return jax.tree.map(_reshape, tree)


def get_shardings(tree, axis: int = 0) -> str:
    def _get(x):
        xt = jax.typeof(x)
        return xt.sharding.mesh.axis_names[axis] if xt.sharding else None

    return jax.tree.flatten(jax.tree.map(_get, tree))[0]


def _vjp_batched_no_sharding(
    fun, diff_argnums, batch_argnums, batch_size, return_forward, primals, vector_arg
):
    batch_args, *reconstruct_primals = split_tuple(primals, batch_argnums)

    def compute_batch_vjp(batch):
        batch_arg_chunk, vector_chunk = batch
        primals_chunk = recompose_tuple(batch_arg_chunk, *reconstruct_primals)
        diff_args, *reconstruct_args = split_tuple(primals_chunk, diff_argnums)

        def _fun(diff_args):
            return fun(*recompose_tuple(diff_args, *reconstruct_args))

        fwd, vjp_fun = nkvjp(_fun, diff_args) # TODO: maybe jax.vjp?
        if return_forward:
            return fwd, vjp_fun(vector_chunk)[0]
        else:
            return vjp_fun(vector_chunk)[0]

    return reduce(
        compute_batch_vjp,
        (batch_args, vector_arg),
        batch_size=batch_size,
        stack_first_output=return_forward,
    )


def _vjp_batched_sharding(
    fun, diff_argnums, batch_argnums, batch_size, return_forward, primals, vector_arg
):
    primals_diff, *reconstruct_primals = split_tuple(primals, diff_argnums)
    batch_args, *_ = split_tuple(primals, batch_argnums)
    sharded_axis_name = jax.typeof(jax.tree.flatten(batch_args)[0][0]).sharding.spec[0]
    mesh = jax.typeof(jax.tree.flatten(batch_args)[0][0]).sharding.mesh
    primals_diff_replicated = pvarying_replicated(primals_diff, sharded_axis_name)
    primals_replicated = recompose_tuple(primals_diff_replicated, *reconstruct_primals)

    in_specs = jax.tree.map(
        lambda x: jax.typeof(x).sharding.spec, (primals_replicated, vector_arg)
    )
    out_specs = jax.tree.map(
        lambda x: jax.typeof(x).sharding.spec, primals_diff_replicated
    )
    if return_forward:
        result_aval = jax.eval_shape(fun, *primals)
        out_specs = (
            jax.tree.map(lambda x: jax.typeof(x).sharding.spec, result_aval),
        ) + out_specs

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )
    def vjp_scan(primals_shard, vector_shard):
        batch_args, *reconstruct_primals = split_tuple(primals_shard, batch_argnums)

        def compute_batch_vjp(batch):
            batch_arg_chunk, vector_chunk = batch
            primals_chunk = recompose_tuple(batch_arg_chunk, *reconstruct_primals)
            diff_args, *reconstruct_args = split_tuple(primals_chunk, diff_argnums)

            def _fun(diff_args):
                return fun(*recompose_tuple(diff_args, *reconstruct_args))

            fwd, vjp_fun = nkvjp(_fun, diff_args)  # TODO: maybe jax.vjp?
            (vjp_batch,) = vjp_fun(vector_chunk)
            if return_forward:
                return fwd, vjp_batch
            else:
                return vjp_batch

        res = reduce(
            compute_batch_vjp,
            (batch_args, vector_shard),
            batch_size=batch_size,
            stack_first_output=return_forward,
            init_fun=lambda x: jax.lax.pvary(jnp.zeros_like(x), sharded_axis_name),
        )
        return res

    if return_forward:
        fwd_shards, vjp_shards = vjp_scan(primals_replicated, vector_arg)
        fwd = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), fwd_shards)
    else:
        vjp_shards = vjp_scan(primals_replicated, vector_arg)

    vjp_vals = stack_shards(vjp_shards, axis=0)
    vjp_vals = jax.tree.map(lambda x: jnp.sum(x, axis=0), vjp_vals)

    if return_forward:
        return fwd, vjp_vals
    else:
        return vjp_vals


def vjp(
    fun,
    *primals,
    argnums=(),
    batch_argnums=(),
    batch_size=None,
    return_forward: bool = False,
):
    batch_args, nonbach_args, idxs = split_tuple(primals, batch_argnums)
    batch_args_flat, _ = jax.tree.flatten(batch_args)
    batch_args_shardings = [jax.typeof(x).sharding for x in batch_args_flat]
    if all(
        isinstance(x, jax.sharding.SingleDeviceSharding) for x in batch_args_shardings
    ) or all(x.is_fully_replicated for x in batch_args_shardings):
        return jax.tree_util.Partial(
            HashablePartial(
                _vjp_batched_no_sharding,
                fun,
                argnums,
                batch_argnums,
                batch_size,
                return_forward,
            ),
            primals,
        )
    else:
        return jax.tree_util.Partial(
            HashablePartial(
                _vjp_batched_sharding,
                fun,
                argnums,
                batch_argnums,
                batch_size,
                return_forward,
            ),
            primals,
        )
