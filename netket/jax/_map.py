from functools import partial, wraps


import jax
import jax.numpy as jnp
from jax.sharding import SingleDeviceSharding, NamedSharding, Mesh, PartitionSpec as P

@wraps(jax.lax.map)
def map(f, x, batch_size: int | None = None):
    # check sharding
    if not jax.sharding.get_abstract_mesh()._any_axis_explicit:
        return jax.lax.map(f, *x, batch_size=batch_size)
    else:
        x_aval = jax.typeof(x)
        x_sharding = x.sharding
        if isinstance(x_sharding, SingleDeviceSharding):
            # If the input is sharded on a single device, we can use lax.map directly
            return jax.lax.map(f, *x, batch_size=batch_size)
        elif isinstance(x_sharding, NamedSharding):
            if x_sharding.spec[0] is None:
                # If the first axis is not sharded, we can use lax.map directly
                return jax.lax.map(f, *x, batch_size=batch_size)
            else:
                mesh = x_sharding.mesh
                n_devices_on_sharded_axis = mesh.shape[x_sharding.spec[0]]
                x_reshape = jnp.reshape(x, (n_devices_on_sharded_axis,) + x_sharding.shard_shape(x.shape))
                x_reshape = jnp.transpose(x_reshape, (1,0) + tuple(range(2, x.ndim+1)))
                result = jax.lax.map(f, x_reshape, batch_size=batch_size)
                def _reshape_result(y):
                    # Reshape the result back to the original shape
                    y_t = jnp.transpose(y, (1, 0) + tuple(range(2, y.ndim)))
                    return jnp.reshape(y_t, (-1,) + y.shape[2:])
                return jax.tree.map(_reshape_result, result)
        else:
            raise NotImplementedError(
                f"Unsupported sharding type: {type(x_sharding)}. "
                "Only SingleDeviceSharding and NamedSharding are supported."
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