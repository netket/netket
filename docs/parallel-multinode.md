# Multiple nodes (GPUs)

## Setting up JAX distributed

:::{warning}
**Important**: The JAX distributed setup details may change over time. This documentation was written in July 2025.
:::

For multi-node calculations, initialize JAX's distributed computing at the beginning of your script. The function {func}`jax.distributed.initialize` (see [documentation](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html)) works automatically with SLURM, but if you're not using SLURM, there are many parameters that need to be specified manually.

Always verify that JAX correctly detects all processes and devices by adding print statements as shown below. This is extremely useful to ensure that your code is being distributed correctly and to debug possible deadlocks in the initialize function. As a general suggestion, running the script below can be useful to verify that everything works correctly.

```python
import jax

# Initialize distributed JAX (works automatically with SLURM)
jax.distributed.initialize()

# Always print this to verify correct setup
print(f"[{jax.process_index()}/{jax.process_count()}] devices:", jax.devices(), flush=True)
print(f"[{jax.process_index()}/{jax.process_count()}] local devices:", jax.local_devices(), flush=True)
print(f"I will be running a calculation among {jax.process_count()} tasks, "
      f"using a total of {len(jax.devices())} devices "
      f"({len(jax.local_devices())} per slurm task). "
      f"If this does not match your expected number of total devices, "
      f"something is misconfigured", flush=True)

import netket as nk
# ... rest of your code
```

:::{warning} Checking that the setup is correct

A common mistake is that different nodes do not communicate with each other, resulting in your calculation running independently on each node instead of as a single distributed computation.

To verify that your distributed setup is working correctly, check that:

- The code above prints `[i/N]` for each process, where `i` goes from 0 to N-1 and N is the total number of processes across all nodes
- `jax.local_devices()` shows the GPU(s) visible to each individual process
- `jax.devices()` shows all GPUs across all nodes (total should be N_gpus_per_node × N_nodes)
- The diagnostic message shows the expected total number of devices for your cluster configuration
:::

## SLURM job script example

When using {func}`jax.distributed.initialize()` with multiple nodes, JAX assumes there is 1 task per GPU. Therefore, you must set `--ntasks-per-node` equal to the number of GPUs per node. In the example above, we assume nodes with 4 GPUs.

The `--gres=gpu:N` option works correctly with JAX's automatic detection, but `--gpus-per-task` does not work and should be avoided.
Here's a typical SLURM script for running NetKet across multiple nodes:

```bash
#!/bin/bash
#SBATCH --job-name=netket-distributed
#SBATCH --output=netket_%j.txt
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=4         # Tasks per node (must equal GPUs per node for jax.distributed.initialize())
#SBATCH --cpus-per-task=5           # CPU cores per task
#SBATCH --gres=gpu:4                # GPUs per node
#SBATCH --time=02:00:00

module purge

# Force JAX to use GPUs, and fail if he cannot use them.
export JAX_PLATFORM_NAME=gpu

# Launch with srun (SLURM's parallel launcher)
srun uv run python your_netket_script.py
```

It is possible to use different SLURM configurations, but they must be configured manually. For more details, refer to {func}`jax.distributed.initialize` and the [JAX clusters folder](https://github.com/google/jax/tree/main/jax/_src/clusters) in the JAX repository.

Note that jax automatically installs a copy of CUDA in the python environment, so you don't need to install your own or load the relevant modules.
This way, CUDA is always updated and you are sure of using the correct version.
It is best to avoid loading the cluster-provided cuda versions.

## Working with distributed arrays

When working with distributed computing, special care is needed for I/O operations and array printing.

### Safe printing of distributed arrays

Distributed arrays need to be replicated before printing to avoid errors:

```python
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

# Example distributed array
mesh = jax.distributed.get_abstract_mesh()
sharded_spec = NamedSharding(mesh, jax.sharding.PartitionSpec('S'))
replicated_spec = NamedSharding(mesh, jax.sharding.PartitionSpec())

distributed_array = jax.device_put(jnp.ones((100, 50)), sharded_spec)

# WRONG: This may cause errors or incomplete output
# print(distributed_array)

# CORRECT: Replicate before printing
replicated_array = jax.lax.with_sharding_constraint(
    distributed_array, replicated_spec
)
if jax.process_index() == 0:
    print("Array shape:", replicated_array.shape)
    print("Array values:", replicated_array)
```

### Safe file I/O

:::{note}
NetKet's built-in loggers handle distributed I/O correctly automatically. The pattern below is needed only when writing custom I/O code.
:::

When saving data, ensure only one process writes to avoid conflicts:

```python
import jax
import jax.numpy as jnp
import numpy as np

def save_distributed_array(array, filename):
    """Safely save a distributed array to file"""
    # First unshard the data by getting it from devices
    array_local = jax.device_get(array)
    # Then convert to numpy on all processes
    array_np = np.array(array_local)
    
    # Only process 0 writes to file
    if jax.process_index() == 0:
        np.save(filename, array_np)
        print(f"Saved array to {filename}", flush=True)

# Example usage
distributed_result = some_netket_calculation()
save_distributed_array(distributed_result, "results.npy")
```

### Testing distributed code locally

NetKet provides the `djaxrun` utility for testing multi-process distributed code locally. This simulates multi-node execution and is **strongly recommended** for testing scripts before submitting to HPC clusters:

```bash
# Test with 2 processes (simulating 2 nodes)
djaxrun --simple -np 2 python3 your_script.py

# Test with 4 processes
djaxrun --simple -np 4 python3 your_script.py
```

The `djaxrun` utility works well with CPUs but requires careful GPU management for multi-GPU testing:

```python
# For GPU testing, ensure each process sees only one GPU
import os
import jax

# Set GPU visibility based on process rank
if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

jax.distributed.initialize()
print(f"Process {jax.process_index()}: {jax.local_devices()}")

import netket as nk
# ... your NetKet code
```


## Troubleshooting

### Common issues

- **No output or deadlock**: If you don't see any messages and no prints of local devices, `jax.distributed.initialize()` is likely deadlocked because nodes cannot communicate. This happens on some clusters due to security restrictions blocking communication among compute nodes. See the {ref}`grpc_proxy` section in {doc}`sharp-bits` for solutions.
- **Initialization hangs**: Check that your hostname resolves correctly (`ping $(hostname)`)
- **GPU not detected**: Verify `nvidia-smi` shows GPUs and CUDA is properly installed
- **Slow communication**: Sometimes communication might be very slow. For detailed communication debugging, set `export NCCL_DEBUG=INFO`

## Complete example

Here's a complete example showing distributed NetKet usage:

```python
import jax
import netket as nk

# Initialize distributed computing
jax.distributed.initialize()

# Verify setup
print(f"[{jax.process_index()}/{jax.process_count()}] devices:", jax.devices(), flush=True)
print(f"[{jax.process_index()}/{jax.process_count()}] local devices:", jax.local_devices(), flush=True)
print(f"I will be running a calculation among {jax.process_count()} tasks, "
      f"using a total of {len(jax.devices())} devices "
      f"({len(jax.local_devices())} per slurm task). "
      f"If this does not match your expected number of total devices, "
      f"something is misconfigured", flush=True)

# Define your quantum system
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

# Create Hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# Define neural network model
model = nk.models.RBM(alpha=1)

# Create variational state
vs = nk.vqs.MCState(
    sampler=nk.sampler.MetropolisLocal(hi),
    model=model,
    n_samples=1024
)

# Set up optimizer and driver
opt = nk.optimizer.Sgd(learning_rate=0.01)
gs = nk.VMC(ha, opt, variational_state=vs)

# Run optimization
if jax.process_index() == 0:
    print("Starting VMC optimization...", flush=True)

gs.run(n_iter=300, out='distributed_result')

if jax.process_index() == 0:
    print("Optimization complete!", flush=True)
    print(f"Final energy: {gs.energy}", flush=True)
```

This setup will automatically distribute the computation across all available devices and handle the distributed sampling and optimization efficiently.