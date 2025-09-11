from functools import wraps

import jax
import jax.numpy as jnp
from jax.sharding import SingleDeviceSharding, NamedSharding
from jax.tree_util import tree_flatten, tree_unflatten, tree_map


@wraps(jax.lax.map)
def map(f, xs, *, batch_size: int | None = None):
    """
    Equivalent to jax.lax.map, but supports sharded inputs on the first axis of the
    input pytree.

    .. note::

        This function requires that the first axis of all leaves in the input pytree
        `xs` has the same size AND sharding specification.

        Different sharding specification (such as some leaves are sharded and some
        are not) are not currently implemented and will raise an error. However,
        supporting this case should be straightforward. If needed, please open an
        issue or a PR.
    """
    xs_flat, xs_tree = tree_flatten(xs)

    # -- Input checking --
    try:
        lengths = [x.shape[0] for x in xs_flat]
    except AttributeError as err:
        msg = "map got value with no leading axis to scan over: {}."
        raise ValueError(
            msg.format(", ".join(str(x) for x in xs_flat if not hasattr(x, "shape")))
        ) from err

    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
        msg = "map got values with different leading axis sizes: {}."
        raise ValueError(msg.format(", ".join(str(x.shape[0]) for x in xs_flat)))
    elif len(unique_lengths) == 0:
        msg = "scan got no values to scan over."
        raise ValueError(msg)

    # If no batch size is specified, just use jax.vmap which ensures the simplest
    # semantics of mapping over the first axis of the input pytree and supports sharding.
    if batch_size is None:
        return jax.vmap(f)(xs)

    # Get each leaf's aval and sharding
    xs_avals = [jax.typeof(leaf) for leaf in xs_flat]
    xs_shardings = [aval.sharding for aval in xs_avals]

    # If no explicit axes in the current mesh, or all inputs reside on a single device,
    # jax.lax.map will work out of the box.
    if not jax.sharding.get_abstract_mesh()._any_axis_explicit or all(
        isinstance(s, SingleDeviceSharding) for s in xs_shardings
    ):
        return jax.lax.map(f, xs, batch_size=batch_size)

    # Mixed sharding types: some Named but not all
    if any(isinstance(s, NamedSharding) for s in xs_shardings) and not all(
        isinstance(s, NamedSharding) for s in xs_shardings
    ):
        raise NotImplementedError(
            "Mixed sharding types: some inputs are sharded while others are not. "
            "Please shard all inputs the same."
        )

    # Case 2: all NamedSharding → check and peel off the first‐axis shard
    if all(isinstance(s, NamedSharding) for s in xs_shardings):
        # If inputs are sharded like (D*N@S, ...) whree D is the number of devices,
        # we reshape 'n swap to (N, D@S, ...) and then scan over the first axis,
        # vmapping over the second.

        specs0 = [s.spec[0] for s in xs_shardings]
        # if *none* of them shard the first axis, direct
        if all(sp is None for sp in specs0):
            return jax.lax.map(f, xs, batch_size=batch_size)
        # require *all* to shard the same named axis
        if any(sp is None for sp in specs0) or len({*specs0}) != 1:
            raise ValueError(
                f"Inconsistent first‐axis sharding across pytree: {specs0}"
            )
        axis_name = specs0[0]
        mesh = xs_shardings[0].mesh
        # find which mesh‐axis index it is
        n_devs = mesh.shape[axis_name]

        # reshape + transpose helper
        def peel_and_move(leaf, sh):
            # leaf.shape = (B, *rest), where B = n_devs * local_batch
            local_shape = sh.shard_shape(leaf.shape)
            # first reshape → (n_devs, local_batch, *rest)
            y = jnp.reshape(leaf, (n_devs,) + tuple(local_shape))
            # then bring the local_batch in front → (local_batch, n_devs, *rest)
            y = jnp.transpose(y, (1, 0) + tuple(range(2, y.ndim)))
            return y

        # apply to every leaf
        xs_peeled = [peel_and_move(leaf, sh) for leaf, sh in zip(xs_flat, xs_shardings)]

        xs_tr = tree_unflatten(xs_tree, xs_peeled)
        # vmap over the extra axis to emulate lax.map semantics
        mapped = jax.lax.map(
            jax.vmap(f),
            xs_tr,
            # bug in jax/#29867
            batch_size=min(batch_size, jax.tree.leaves(xs_tr)[0].shape[0]),
        )

        # inverse reshape+transpose helper
        def reassemble(y):
            # y.shape = (local_batch, n_devs, *rest)
            # undo transpose → (n_devs, local_batch, *rest)
            y2 = jnp.transpose(y, (1, 0) + tuple(range(2, y.ndim)))
            # flatten back → (B, *rest)
            return jnp.reshape(y2, (-1,) + y2.shape[2:])

        res = tree_map(reassemble, mapped)
        return res

    # anything else is unsupported
    raise NotImplementedError(
        f"Unsupported sharding types: {set(type(s) for s in xs_shardings)}"
    )


# import numpy as np
# import jax
# import jax.numpy as jnp
# from jax.sharding import Mesh, PartitionSpec as P, AxisType, reshard

# # Setup: 2 CPU devices
# jax.config.update("jax_num_cpu_devices", 2)
# devices = np.array(jax.devices())
# mesh = jax.make_mesh((2,),("s",), axis_types=(AxisType.Explicit,),)
# jax.sharding.set_mesh(mesh) # Set this as the default mesh for jax.

# def simple_func(w, x):
#     return jnp.sum(w * x, axis=-1)

# # Make inputs
# w = jnp.array([1.0, 2.0, 3.0, 4.0])
# x = jnp.ones((10, 4))

# # Setup sharding
# xs = reshard(x, P("s", None))

# r1=jax.lax.map(lambda _x: simple_func(w, _x), x, batch_size=2)
# r2= custom_map(lambda _x: simple_func(w, _x), xs, batch_size=2)
