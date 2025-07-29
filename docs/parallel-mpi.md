# Multiple CPUs (MPI)

:::{warning}
**Experimental feature**: This MPI backend is experimental and **CPU-only** - it should not be used with GPUs. For GPU-based distributed computing, use the standard JAX distributed setup described in the [multi-node multi-GPU guide](parallel-multinode.md). Please let us know if this works for your use case.
:::

The MPI backend can be used as an alternative to GLOO for multi-process CPU communication across single or multiple nodes, potentially offering better performance than the default CPU collective implementations. This is useful when you need to run distributed NetKet calculations on CPU-only clusters.

**Use the MPI backend when:**

- You need distributed computing across CPU-only nodes (single or multiple nodes)
- You want MPI-level performance for CPU communication
- GPUs are not available or not desired

## Installation

You need to have MPI installed on your system. **Note**: OpenMPI 5 is not supported, so on macOS use `mpich` instead.

For detailed installation instructions, see the [mpibackend4jax repository](https://github.com/mpi4jax/mpibackend4jax).

```bash
# Using uv (recommended)
uv add git+https://github.com/mpi4jax/mpibackend4jax

# Or using pip
pip install git+https://github.com/mpi4jax/mpibackend4jax
```

:::{details} Detailed Installation Instructions
:class: dropdown

If you need help with MPI installation:

**On Ubuntu/Debian:**

```bash
sudo apt-get install libmpich-dev mpich
```

**On macOS:**

```bash
brew install mpich
```

Then install mpibackend4jax as shown above.
:::

## Usage with NetKet

```python
# Import mpibackend4jax BEFORE importing JAX
import mpibackend4jax as _mpi4jax  # noqa: F401

import jax
import netket as nk

# Initialize distributed computing
jax.distributed.initialize()

# Verify setup
print(f"[{jax.process_index()}/{jax.process_count()}] MPI setup complete", flush=True)
print(f"[{jax.process_index()}/{jax.process_count()}] Devices: {jax.local_devices()}", flush=True)

# Define quantum system
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
    print("Starting MPI-based VMC optimization...", flush=True)

gs.run(n_iter=100, out='mpi_result')

if jax.process_index() == 0:
    print("Optimization complete!", flush=True)
```

## Running with MPI

Launch your script using the MPI launcher:

```bash
# Single node with 4 processes
mpirun -n 4 python your_netket_script.py

# Multiple nodes (example with 2 nodes, 4 processes per node)
mpirun -np 8 --hostfile hostfile python your_netket_script.py
```