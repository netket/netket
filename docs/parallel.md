# Parallel Computing

NetKet supports single-node multi-gpu and multi-node distributed computing through JAX's native sharding interface, which makes working with arrays distributed across multiple devices easier.
Moreover, multi-node setups make use of jax's [multi-host controller logic](https://docs.jax.dev/en/latest/multi_process.html).
This page provides an overview of NetKet's distributed computing capabilities and links to detailed guides for specific use cases.

:::{important}
**Essential reading**: Understanding JAX's distributed computing model is crucial for effective use of NetKet's distributed capabilities. Please read these JAX documentation resources:

- [JAX distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [JAX multi-process documentation](https://docs.jax.dev/en/latest/multi_process.html)
:::

:::{note}
**Historical note**: Previous versions of NetKet required setting `NETKET_EXPERIMENTAL_SHARDING=1`. This is now enabled by default. Earlier versions also supported MPI-based parallelization, which has been removed in favor of JAX's superior sharding capabilities.
:::

## Overview of parallelization methods

NetKet provides three approaches for distributed and parallel computing:

### [1. Single-node multi-GPU](parallel-multigpu.md)
For most single-node multi-GPU usage, no special configuration is needed - NetKet automatically uses all visible GPUs on your system. This is the easiest way to scale up your calculations.
Nevertheless, some care must be taken when working with arrays that may live across multiple GPUs, so do give a read to this.

### [2. Multi-node multi-GPU (HPC clusters)](parallel-multinode.md)
For calculations spanning multiple nodes.
This uses jax's multi-controller setup.
To understand how to use it, you should know how the single-node multi-gpu works and how to work with sharded arrays.

### [3. Multi-CPU with MPI](parallel-mpi.md)
Jax native multi-cpu optimizations are terrible.
If you want to make better use of the many cores of your laptop/workstation, or if you want to distribute your calculations among many nodes, have a look here.
This is an experimental CPU-only approach using the same sharding and multi-controller setup as above (so you should read their instructions) but uses an MPI backend instead of NCCL for communication.


```{toctree}
:hidden:

parallel-multigpu
parallel-multinode
parallel-mpi
```
