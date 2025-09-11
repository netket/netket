from functools import partial
from typing import Union, Iterable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

from netket.utils.jax import HashablePartial

from ._reduce import reduce
from ._vjp import vjp as nkvjp


def split_tuple(
    data: tuple, indices: Union[int, Iterable[int]]
) -> tuple[tuple, tuple, tuple[int, ...]]:
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
    selected: tuple,
    remainder: tuple,
    indices: Iterable[int],
) -> tuple:
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
            and xt.ndim > 0
            and xt.sharding.spec[0] == axis_name
        ):
            return x
        x_shard = jnp.repeat(x[None, ...], n_devices, axis=0)  # shape (num_batches, M)
        x_shard = reshard(x_shard, P(axis_name, *(None for _ in range(x.ndim))))
        return x_shard

    def _must_reshape(x):
        xt = jax.typeof(x)
        if (
            isinstance(xt.sharding, jax.sharding.NamedSharding)
            and xt.ndim > 0
            and xt.sharding.spec[0] == axis_name
        ):
            return False
        return True

    return jax.tree.map(_pvarying_replicated, tree), jax.tree.map(_must_reshape, tree)


def drop_leading_dim(tree, tree_reshape):
    axis = 0

    def _drop(x, reshape: bool):
        if not reshape:
            return x
        assert x.ndim > 0, "Cannot drop leading dimension of a scalar"
        assert (
            x.shape[axis] == 1
        ), f"Cannot drop leading dimension of a non-singleton axis {x.shape}"
        if x.ndim == 1:
            return x.reshape(())
        return x.reshape(*x.shape[:axis], *x.shape[axis + 1 :])

    return jax.tree.map(_drop, tree, tree_reshape)


def add_leading_dim(tree, tree_reshape):
    axis = 0

    def _drop(x, reshape: bool):
        if not reshape:
            return x
        return jnp.expand_dims(x, axis=axis)

    return jax.tree.map(_drop, tree, tree_reshape)


def reduce_shards(tree, tree_reshape):
    """
    Reduces the leading dimension of the tree, which is assumed to be the sharded axis.
    This is useful for accumulating results from multiple devices.
    """

    def _reduce(x, reshape: bool):
        if not reshape:
            return x
        return jnp.sum(x, axis=0)

    return jax.tree.map(_reduce, tree, tree_reshape)


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
    fun,
    diff_argnums,
    batch_argnums,
    batch_size,
    return_forward,
    rc_vjp_is_complex,
    primals,
    vector_arg,
):
    # split the primals that must be scanned over from those that we do not scan over
    batch_args, *reconstruct_primals = split_tuple(primals, batch_argnums)
    # The arguments that are both to be diffed and to be batched will have to be stacked
    # along the first axis, so we need to stack them.
    # ADditionally, if we are to return the forward pass we will need to stack it as well
    stack_outnums = tuple(sorted(set(diff_argnums) & set(batch_argnums)))
    if return_forward:
        stack_outnums = (0,) + tuple(s + 1 for s in stack_outnums)

    def compute_batch_vjp(args):
        batch_arg_batch, vector_chunk = args
        primals_batch = recompose_tuple(batch_arg_batch, *reconstruct_primals)

        # As we do not differentiate wrt some arguments, we define the function
        # to differentiate as a function of the remaining arguments only.
        diff_args, *reconstruct_args = split_tuple(primals_batch, diff_argnums)

        def _fun(diff_args):
            _args = recompose_tuple(diff_args, *reconstruct_args)
            return fun(*_args)

        # TODO: maybe jax.vjp?
        vjp_impl = nkvjp if rc_vjp_is_complex else jax.vjp
        fwd, vjp_fun = vjp_impl(_fun, diff_args)

        # If we are returning the forward pass, we need to unpack the vjp otherwise we cannot select the stack outnums correctly.
        if return_forward:
            res = (fwd,) + vjp_fun(vector_chunk)[0]
        else:
            res = vjp_fun(vector_chunk)[0]
        return res

    # This performs a mapreduce over the batches of the batch_args and the vector_arg.
    # This reduce does not work with sharded arrays along the map axis.
    output = reduce(
        compute_batch_vjp,
        (batch_args, vector_arg),
        batch_size=batch_size,
        stack_outnums=stack_outnums,
    )
    if return_forward:
        out_fwd, *out_vjp = output
        return out_fwd, tuple(out_vjp)
    else:
        return output


def _vjp_batched_sharding(
    fun,
    diff_argnums,
    batch_argnums,
    batch_size,
    return_forward,
    rc_vjp_is_complex,
    primals,
    vector_arg,
):
    # Extract the mesh and the sharded axis name for the batched primals.
    batch_args, *_ = split_tuple(primals, batch_argnums)
    mesh = jax.typeof(jax.tree.flatten(batch_args)[0][0]).sharding.mesh
    sharded_axis_name = jax.typeof(jax.tree.flatten(batch_args)[0][0]).sharding.spec[0]

    # The arguments that are both to be diffed and to be batched will have to be stacked
    # along the first axis, so we need to stack them.
    # ADditionally, if we are to return the forward pass we will need to stack it as well
    stack_outnums = tuple(sorted(set(diff_argnums) & set(batch_argnums)))
    if return_forward:
        stack_outnums = (0,) + tuple(s + 1 for s in stack_outnums)

    # We need to transform the diffed primals from f32[P] to f32[d*P@S], where
    # S is the sharded axis and d is the number of devices. This allows us to rewrite the
    # vjp in the shard_map without introducing an all-gather, because every device computes
    # its own contribution to the vjp.
    # From limited testing, it seems that jax is smart enough to just do a copy here
    primals_diff, *reconstruct_primals = split_tuple(primals, diff_argnums)
    primals_diff_replicated, primals_diff_struct = pvarying_replicated(
        primals_diff, sharded_axis_name
    )
    # Primals_replicated is primals, where the diffed primals are now 'bigger' and sharded
    # (technically, this shoudl be a no-op).
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
        ) + (out_specs,)

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        axis_names={sharded_axis_name},
    )
    def vjp_scan(primals_shard, vector_shard):
        # we have an extra dimension on the diffable primals
        _primals_diff, *_reconstruct_primals = split_tuple(primals_shard, diff_argnums)
        _primals_diff = drop_leading_dim(_primals_diff, primals_diff_struct)
        primals_shard = recompose_tuple(_primals_diff, *_reconstruct_primals)

        batch_args, *reconstruct_primals = split_tuple(primals_shard, batch_argnums)

        def compute_batch_vjp(batch):
            batch_arg_chunk, vector_chunk = batch
            primals_chunk = recompose_tuple(batch_arg_chunk, *reconstruct_primals)
            diff_args, *reconstruct_args = split_tuple(primals_chunk, diff_argnums)

            def _fun(diff_args):
                return fun(*recompose_tuple(diff_args, *reconstruct_args))

            # TODO: maybe jax.vjp?
            vjp_impl = nkvjp if rc_vjp_is_complex else jax.vjp
            fwd, vjp_fun = vjp_impl(_fun, diff_args)
            (vjp_batch,) = vjp_fun(vector_chunk)
            vjp_batch = add_leading_dim(vjp_batch, primals_diff_struct)
            if return_forward:
                return (fwd,) + vjp_batch
            else:
                return vjp_batch

        res = reduce(
            compute_batch_vjp,
            (batch_args, vector_shard),
            batch_size=batch_size,
            stack_outnums=stack_outnums,
            init_fun=jnp.zeros_like,
        )
        if return_forward:
            out_fwd, *out_vjp = res
            res = out_fwd, tuple(out_vjp)
        return res

    if return_forward:
        fwd_shards, vjp_shards = vjp_scan(primals_replicated, vector_arg)
        fwd = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), fwd_shards)
    else:
        vjp_shards = vjp_scan(primals_replicated, vector_arg)

    # Accumulate the shards of the vjps.
    vjp_vals = reduce_shards(vjp_shards, primals_diff_struct)

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
    rc_vjp_is_complex: bool = True,
):
    """
    Computes the vector-Jacobian product (VJP) of a function with respect to (some) of its arguments.

    With no additional arguments, this is equivalent to `jax.vjp`, but it supports one additional feature:
        - batching of the arguments specified in `batch_argnums`, and of the vector argument. This reduces
        the memory footprint of the calculation, as it essentially perform a mapreduce on small batches
        of size `batch_size` (default to infinite) and then reduces the results.

    .. note::

        Because of a limitation of the underlying `shard_map` implementation, the function `fun` must not
        capture any variables that are not in the `primals` tuple.
        To specify which arguments to differentiate with respect to, use the `argnums` argument instead
        of capturing them in a closure.

    Args:
        fun: The function to differentiate.
        *primals: The primal arguments to the function.
        argnums: Which arguments to differentiate with respect to (default: all). This can be a single integer
            or a tuple of integers. If empty, all arguments are differentiated.
        batch_argnums: Which arguments to batch over (default: none). This assumes that the function can be
            safely mapped over those arguments. A strict requirement is that all the batch arguments are
            sharded across the same axis.
        batch_size: The size of the batches to use for batching (default: maximum).
        return_forward: If True, also return the result of the function evaluation (forward pass).
            Default is False.
        rc_vjp_is_complex: If True (default), the VJP of a R->C function will not be projected back to the tangent
            space of the input, but will return the full complex VJP. In practice, this means that the VJP of a complex
            function will return a complex vector, as opposed to the default jax behaviour of returning a real vector.
            This matches the behaviour of `nk.jax.vjp`.

    """
    if isinstance(argnums, tuple) and len(argnums) == 0:
        argnums = tuple(range(len(primals)))
    if isinstance(argnums, int):
        argnums = (argnums,)
    if isinstance(batch_argnums, int):
        batch_argnums = (batch_argnums,)

    batch_args, nonbach_args, idxs = split_tuple(primals, batch_argnums)
    batch_args_flat, _ = jax.tree.flatten(batch_args)
    batch_args_shardings = [jax.typeof(x).sharding for x in batch_args_flat]

    # If we have all SingleDeviceSharding/are fully replicated, we can use a simpler implementation
    # that does not use shard_map.
    # TODO: See https://github.com/jax-ml/jax/issues/29647 to simplify the fully replicated condition.
    if all(
        isinstance(x, jax.sharding.SingleDeviceSharding) for x in batch_args_shardings
    ) or (
        all(x.is_fully_replicated for x in batch_args_shardings)
        and all(x.spec[0] is None for x in batch_args_shardings)
    ):
        return jax.tree_util.Partial(
            HashablePartial(
                _vjp_batched_no_sharding,
                fun,
                argnums,
                batch_argnums,
                batch_size,
                return_forward,
                rc_vjp_is_complex,
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
                rc_vjp_is_complex,
            ),
            primals,
        )
