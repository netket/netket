# Parallelization

NetKet normally only uses the jax default device `jax.local_devices()[0]` to perform calculations, and ignores the others. This means that if you want to fully exploit your many CPU cores or several GPUs, you must resort to one of two parallelization strategies: MPI or Sharding.

- (MPI, Stable) Explicit parallelization by distributing the markov chains and samples across multiples nodes/devices. This is achieved by using [MPI (with mpi4jax)](mpi). When using MPI, netket/jax will only use the jax default device `jax.local_devices()[0]` on every rank, and you must ensure that this corresponds to different devices (either cores or GPUs).
- (Sharding, Experimental) [native collective communication built into jax](sharding) is jax's preferred mode of distributing calculations, and is discussed in [Jax Distributed Computation tutorial](https://jax.readthedocs.io/en/latest/multi_process.html). This mode can be used both on a single node with many GPUs or many nodes with many GPUs.

:::{warning}
### What should you use?

Getting MPI up and running on a SLURM HPC cluster can be complicated, and sharding is incredibly easy to setup: install jax and you are done!
    
However, sharding works well only for GPUs, and CPU support is an afterthought that performs terribly, and we mainly only use it for testing locally that your code will run before sending it to the cluster.

Sharding code is also much simpler to write and maintain for us, so in the future it will be the preferred mode. Be careful that some operators based on Numba do not work with sharding, but they can all be converted to a version that works well with it.
:::

NetKet is written such that code that runs with sharding will also work with MPI, and vice-versa. The main thing you should
be careful is when you save files to do so only on the master rank.

(mpi)=
## MPI (mpi4jax)

Requires that {code}`mpi4py` and {code}`mpi4jax` are installed, please refer to [Installation#MPI](install_mpi).

When using {code}`netket` it is crucial to run Python with the same implementation and version of MPI that the {code}`mpi4py` module is compiled against.
If you encounter issues, you can check whether your MPI environment is set up properly by running:

```
$ mpirun -np 2 python3 -m netket.tools.check_mpi
mpi4py_available             : True
mpi4jax_available            : True
available_cpus (rank 0)       : 12
n_nodes                      : 1
mpi4py | MPI version         : (3, 1)
mpi4py | MPI library_version : Open MPI v4.1.0, package: Open MPI brew@BigSur Distribution, ident: 4.1.0,  repo rev: v4.1.0, Dec 18, 2020
```

This should print some basic information about the MPI installation and, in particular, pick up the correct `n_nodes`.
If you get the same output multiple times, each with {code}`n_nodes : 1`, this is a clear sign that your MPI setup is broken.
The tool above also reports the number of (logical) CPUs that might be subscribed by Jax on every independent MPI rank during linear algebra operations.
Be mindfull that Jax, in general, is like an invasive plant and tends to use all resources that he can access, and
the environment variables above might not prevent it from making use of the `available_cpus`.
On Mac it is not possible to control this number.
On Linux it can be controlled using `taskset` or `--bind-to core` when using `mpirun`.


(sharding)=
## Sharding (Native Jax parallelism)

Historically the principal way to run {code}`netket` in parallel has been to use MPI via {code}`mpi4py` and {code}`mpi4jax`.
However, recently jax gained support for shared arrays and collective operations on multiple devices/nodes (see [here](https://jax.readthedocs.io/en/latest/jax_array_migration.html#jax-array-migration) and [here](https://jax.readthedocs.io/en/latest/multi_process.html)) and we adapted {code}`netket` to support those, enabling native parallelism via jax.

:::{warning}
This feature is still a work in progress, but as of September 2024 it is very reliable and we are routinely using it
for our research.

Moreover, we found that sharding leads to a consistent 5-10% speedup over MPI when using multiple GPUs.
:::

(jax_single_process)=
### Sharding: Single node, multiple GPUs (single process sharding)

To run on a single process with multiple devices on a single node usually all that is necessary is to set the environment flag `NETKET_EXPERIMENTAL_SHARDING=1`, e.g. by setting them before importing {code}`netket`.

As this mode is having a single python process control all gpus, code is easy to write and you don't need to take care of
anything in particular.
- __GPU__
```python
import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
# ...
```
- __CPU__

You can force jax to use multiple threads as cpu devices (see [jax 101](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html#aside-hosts-and-devices-in-jax)), e.g.:
```python
import os
os.environ['NETKET_EXPERIMENTAL_SHARDING_CPU'] = '8'

import netket as nk
# ...
```
You should only use this to test things that they work, but not for anything serious.

(jax_multi_process)=
### Sharding: Multiple nodes

Background:
_Jax_ internally uses the [grpc library](https://grpc.io) (launching a http server) for setup and book-keeping of the cluster and the [nvidia nccl library](https://developer.nvidia.com/nccl) for communication between gpus, and (experimentally) MPI or [gloo](https://github.com/facebookincubator/gloo) for communication between cpus.

To launch netket on a multi-node cluster usually all that is required is to add a call to `jax.distributed.initialize()` at the top of the main script, see the follwing examples.
These scripts can be conveniently launched with `srun` (on slurm clusters) or `mpirun`.
For more details and manual setups we refer to the [jax documentation](https://jax.readthedocs.io/en/latest/multi_process.html).

By default, on slurm clusters, jax will see a single GPU per process so if you have 4 GPUs per node, you should launch 4 tasks per node.
Give a look at the [Cluster Sharding setup](cluster-sharding-setup) for an example of using SLURM with this.

```python
import jax

# This assumes you have mpi4py installed, but is very reliable.
jax.distributed.initialize(cluster_detection_method="mpi4py")

import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
# ...
```
It is required that `libnccl2` and `libnccl2-dev` are installed in addition to cuda. If you run into communication errors, you might want to set the environment variable `NCCL_DEBUG=INFO` for detailed error messages.


### Sharding: Testing locally

To test sharding locally on a computer you can write a script that begin like the following

```python
import jax

# This assumes you have mpi4py installed, but is very reliable.
jax.config.update("jax_cpu_collectives_implementation", "gloo")
jax.distributed.initialize(cluster_detection_method="mpi4py")

import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
# ...

if jax.process_index() == 0:
    print("only printed from rank 0")
```

and then you launch it with the command

```bash
mpirun -np 2 python yourscript.py
```

This will run netket with sharding mode, and will only use MPI to launch multiple copies of python
and to setup the communication among different ranks.

Do note that for CPUs this will be very slow. While if you have multiple GPUs, it will default to using
a single GPU per process.


#### MPITrampoline backend (very experimental)
Experimental, requires `jax/jaxlib>=0.4.27`.

- Download and compile [MPIwrapper](https://github.com/eschnett/MPIwrapper)

```bash
git clone https://github.com/eschnett/MPIwrapper.git
cd MPIwrapper
mkdir build
cd build
cmake ../
make
```
 The `libmpiwrapper.so` can be found in the `build` folder created above.

```python
import os
os.environ['MPITRAMPOLINE_LIB'] = '/path/to/libmpiwrapper.so'

import jax
jax.config.update('jax_cpu_collectives_implementation', 'mpi')
jax.distributed.initialize()

os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
# ...
```


(grpc_proxy)=
### GRPC incompatibility with http proxy wildcards
We noticed that communication errors can arise when a http proxy is used on the EPFL cluster. Grpc will try to communicate with the other nodes via the proxy, whenever they are only excluded in the `no_proxy` variable via wildcards (e.g. `no_proxy=10.0.0.*`) which we found grpc cannot parse. To avoid this one needs to include all addresses explicitly.

Alternatively, a simple way to work around it is to disable the proxy completely for jax by unsetting the respective environment variables (see [grpc docs](https://grpc.github.io/grpc/cpp/md_doc_environment_variables.html)) e.g. as follows:
```python
import os
del os.environ['http_proxy']
del os.environ['https_proxy']
del os.environ['no_proxy']

import jax
jax.distributed.initialize()
```
