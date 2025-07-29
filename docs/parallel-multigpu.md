# Multiple GPUs

NetKet automatically uses all visible GPUs on a single node. For most single-node multi-GPU usage, no special configuration is needed - just run your NetKet code normally and it will scale across all available GPUs.

## Automatic GPU usage

NetKet automatically detects and uses all available GPUs on your system:

```python
import netket as nk
import jax

print("Available devices:", jax.devices())
print("Local devices:", jax.local_devices())

# NetKet will automatically use all visible GPUs for calculations
# No additional configuration needed
```

## Controlling GPU visibility

To use only a subset of available GPUs, set the `CUDA_VISIBLE_DEVICES` environment variable before launching Python:

```bash
# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0
python yourscript.py

# Use GPUs 0 and 2
export CUDA_VISIBLE_DEVICES=0,2
python yourscript.py
```

## Controlling data placement across multiple GPUs

When writing code for multiple GPUs, you may need to explicitly control how arrays are distributed across devices. NetKet handles this internally for its operations, but custom code might require manual sharding:

```python
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

# Create mesh for device placement
mesh = jax.distributed.get_abstract_mesh()

# Example 1: Replicated array (copied to all devices)
array = jnp.ones((1000, 1000))
replicated_sharding = NamedSharding(mesh, jax.P())
replicated_array = jax.device_put(array, replicated_sharding)

# Example 2: Sharded array along first axis (split across devices)
# jax.P('S') means sharding along axis 'S', which by default contains all devices
large_array = jnp.ones((10000, 1000))
sharded_sharding = NamedSharding(mesh, jax.P('S'))
sharded_array = jax.device_put(large_array, sharded_sharding)

# Example 3: Sharded array along second axis
# jax.P(None, 'S') shards along second axis, first axis is not sharded
array_2d = jnp.ones((1000, 8000))
sharded_axis1 = NamedSharding(mesh, jax.P(None, 'S'))
sharded_array_axis1 = jax.device_put(array_2d, sharded_axis1)
```

**Example from NetKet**: The `srt.py` file demonstrates proper sharding techniques for implementing the Stochastic Reconfiguration method with optimal GPU utilization.