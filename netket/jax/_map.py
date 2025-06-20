from functools import wraps

import jax
import jax.numpy as jnp
from jax.sharding import SingleDeviceSharding, NamedSharding
from jax.tree_util import tree_flatten, tree_unflatten, tree_map


@wraps(jax.lax.map)
def map(f, x, batch_size: int | None = None):
    """
    Equivalent to jax.lax.map, but handles arbitrary NamedSharding
    across the first axis, and works on pytrees.
    """
    if batch_size is None:
        return jax.vmap(f)(x)

    # If no explicit axes in the current mesh, just defer to lax.map
    if not jax.sharding.get_abstract_mesh()._any_axis_explicit:
        return jax.lax.map(f, x, batch_size=batch_size)

    # Flatten the pytree
    leaves, treedef = tree_flatten(x)

    # Get each leaf's aval and sharding
    avals = [jax.typeof(leaf) for leaf in leaves]
    shardings = [aval.sharding for aval in avals]

    # Case 1: all SingleDeviceSharding → direct
    if all(isinstance(s, SingleDeviceSharding) for s in shardings):
        return jax.lax.map(f, x, batch_size=batch_size)

    # Mixed sharding types: some Named but not all
    if any(isinstance(s, NamedSharding) for s in shardings) and not all(
        isinstance(s, NamedSharding) for s in shardings
    ):
        raise ValueError(
            "Mixed sharding types: some inputs are sharded while others are not. "
            "Please shard all inputs the same."
        )

    # Case 2: all NamedSharding → check and peel off the first‐axis shard
    if all(isinstance(s, NamedSharding) for s in shardings):
        specs0 = [s.spec[0] for s in shardings]
        # if *none* of them shard the first axis, direct
        if all(sp is None for sp in specs0):
            return jax.lax.map(f, x, batch_size=batch_size)
        # require *all* to shard the same named axis
        if any(sp is None for sp in specs0) or len({*specs0}) != 1:
            raise ValueError(
                f"Inconsistent first‐axis sharding across pytree: {specs0}"
            )
        axis_name = specs0[0]
        mesh = shardings[0].mesh
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
        peeled = [peel_and_move(leaf, sh) for leaf, sh in zip(leaves, shardings)]

        x_tr = tree_unflatten(treedef, peeled)
        # vmap over the extra axis to emulate lax.map semantics
        mapped = jax.lax.map(jax.vmap(f), x_tr, batch_size=batch_size)

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
        f"Unsupported sharding types: {set(type(s) for s in shardings)}"
    )


# import numpy as np
# import jax
# import jax.numpy as jnp
# from jax.sharding import Mesh, PartitionSpec as P, AxisType
# from jax.experimental.shard import reshard, explicit_axes

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
