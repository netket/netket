# Parallelization

NetKet makes use of parallelism in two principal ways:

- By leveraging the just-in-time compilation of XLA vector-instructions are used on CPU (as well as [multiple threads for certain linear algebra operations](xla_multithread)), and, similarly, calculations are parallelized to run on all available cuda cores on GPU.
- Explicit parallelization by distributing the markov chains and samples across multiples nodes/devices. This is achieved by using [MPI (with mpi4jax)](mpi), or alternatively by using [native collective communication built into jax](sharding) (still experimental).

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
## Native Jax parallelism (experimental)

Historically the principal way to run {code}`netket` in parallel has been to use MPI via {code}`mpi4py` and {code}`mpi4jax`.
However, recently jax gained support for shared arrays and collective operations on multiple devices/nodes (see [here](https://jax.readthedocs.io/en/latest/jax_array_migration.html#jax-array-migration) and [here](https://jax.readthedocs.io/en/latest/multi_process.html)) and we adapted {code}`netket` to support those, enabling native parallelism via jax.

:::{warning}
This feature is still experimental and not everything may work perfectly right out of the box.
Any feedback, be it positive or negative, would be greatly appreciated.
:::

(jax_single_process)=
### Single Process

To run on a single process with multiple devices on a single node usually all that is necessary is to set the environment flag `NETKET_EXPERIMENTAL_SHARDING=1`, e.g. by setting them before importing {code}`netket`:
- __GPU__
```python
import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = 1

import netket as nk
# ...
```
- __CPU__

You can force jax to use multiple threads as cpu devices (see [jax 101](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html#aside-hosts-and-devices-in-jax)), e.g.:
```python
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = 1

import netket as nk
# ...
```

(jax_multi_process)=
### Multi-Process

Background:
_Jax_ internally uses the [grpc library](https://grpc.io) (launching a http server) for setup and book-keeping of the cluster and the [nvidia nccl library](https://developer.nvidia.com/nccl) for communication between gpus, and (experimentally) MPI or [gloo](https://github.com/facebookincubator/gloo) for communication between cpus.

To launch netket on a multi-node cluster usually all that is required is to add a call to `jax.distributed.initialize()` at the top of the main script, see the follwing examples.
These scripts can be conveniently launched with `srun` (on slurm clusters) or `mpirun`.
For more details and manual setups we refer to the [jax documentation](https://jax.readthedocs.io/en/latest/multi_process.html).

#### __NCCL (GPU)__
```python
import jax
jax.distributed.initialize()

import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = 1

import netket as nk
# ...
```
It is required that `libnccl2` and `libnccl2-dev` are installed in addition to cuda. If you run into communication errors, you might want to set the environment variable `NCCL_DEBUG=INFO` for detailed error messages.

(multi_device)=
##### Multiple GPU devices per process
According to our testing, it is best to use 1 process per gpu on the cluster.

Nevertheless, if you want to use multiple gpus per process you can force jax to do so by setting `local_device_ids`, e.g. extracting it from `CUDA_VISIBLE_DEVICES` as follows:

```python
import os
import jax
ldi = list(map(int, os.environ.get('CUDA_VISIBLE_DEVICES').split(',')))
jax.distributed.initialize(local_device_ids=ldi)
```

#### MPI (CPU)
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
os.environ['MPITRAMPOLINE_LIB'] = "/path/to/libmpiwrapper.so"

import jax
jax.config.update('jax_cpu_collectives_implementation', 'mpi')
jax.distributed.initialize()

os.environ['NETKET_EXPERIMENTAL_SHARDING'] = 1

import netket as nk
# ...
```

#### GLOO (CPU)
Experimental, requires `jax/jaxlib>=0.4.27`.


```python
import jax
jax.config.update('jax_cpu_collectives_implementation', 'gloo')
jax.distributed.initialize()

import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = 1

import netket as nk
# ...
```

(grpc_proxy)=
### GRPC incompatibility with http proxy wildcards
We noticed that communication errors can arise when a http proxy is used on the cluster. Grpc will try to communicate with the other nodes via the proxy, whenever they are only excluded in the `no_proxy` variable via wildcards (e.g. `no_proxy=10.0.0.*`) which we found grpc cannot parse. To avoid this one needs to include all addresses explicitly.

Alternatively, a simple way to work around it is to disable the proxy completely for jax by unsetting the respective environment variables (see [grpc docs](https://grpc.github.io/grpc/cpp/md_doc_environment_variables.html)) e.g. as follows:
```python
import os
del os.environ['http_proxy']
del os.environ['https_proxy']
del os.environ['no_proxy']

import jax
jax.distributed.initialize()
```
