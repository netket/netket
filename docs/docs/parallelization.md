# Parallelization

NetKet normally only uses the jax default device `jax.local_devices()[0]` to perform calculations, and ignores the others. This means that if you want to fully exploit your many CPU cores or several GPUs, you must resort to a parallelization strategy known as Sharding.

- **Sharding:** [native collective communication built into jax](sharding) is jax's preferred mode of distributing calculations, and is discussed in [Jax Distributed Computation tutorial](https://jax.readthedocs.io/en/latest/multi_process.html). This mode can be used both on a single node with many GPUs or many nodes with many GPUs.

:::{note}
### What should you use?

Sharding is incredibly easy to setup: install jax and you are done!

However, **sharding works well only for GPUs, and CPU support is an afterthought that performs terribly**. For CPU-based workloads, performance will be limited compared to GPU-based sharding.
We mainly only use CPU-based sharding for locally testing that our script will run before sending it to the cluster, but we never use it in production.

Sharding code is also much simpler to write and maintain for us, and is now the only supported parallelization mode. Be careful that some operators based on Numba do not work with sharding, but they can all be converted to a version that works well with it.
:::

**Chef's suggestion:**

|                        | Default |  Sharding  | Sharding + distributed |
|------------------------|---------|------------|------------------------|
| 1 CPU / 1 GPU          |    ✔️    |           |                        |
| 1 Node: MultiCPU       |         |    🐢      |                        |
| 1 Node: MultiGPU       |         |   ✔️       |                        |
| Distributed: CPU       |         |            |            🐢           |
| Distributed: GPU       |         |            |            ✔️           |

Legend:
 - ✔️ Recommended method
 - 🐢 Sharding is slow
 - 🤯 Hard to setup 


(sharding)=
## Sharding (Native Jax parallelism)

NetKet uses JAX's native parallelization capabilities through sharding. JAX provides support for shared arrays and collective operations on multiple devices/nodes (see [here](https://jax.readthedocs.io/en/latest/jax_array_migration.html#jax-array-migration) and [here](https://jax.readthedocs.io/en/latest/multi_process.html)) and NetKet is built to leverage these features for efficient distributed computing.

:::{note}
JAX sharding is NetKet's only supported parallelization mode. It is stable and reliable for production use, providing efficient distributed computing capabilities for both single-node and multi-node configurations.
:::

(jax_single_process)=
### Sharding: Single process, multiple GPUs

If all you want is to use all the GPUs available on your computer for a calculation, all that is necessary is to set the environment flag `NETKET_EXPERIMENTAL_SHARDING=1` **before importing** NetKet. See an example below

As this mode is having a single python process control all gpus, code is easy to write and you don't need to take care of
anything in particular.
- __GPU__
```python
import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
import jax
print("Sharding is enabled:", nk.config.netket_experimental_sharding)
print("The available GPUs are:", jax.devices())
# ...
```
- __CPU__

Sometimes you write your codes on your laptop without GPUs available, to then execute them on clusters with GPUs.
In those cases, it is very handy to be able to run the sharding code on your local computer, using just many CPUs.

You can force jax to 'pretend' that it has multiple CPU devices attached (see [jax 101](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html#aside-hosts-and-devices-in-jax)) by declaring the ``NETKET_EXPERIMENTAL_SHARDING_CPU=Ndevices`` environment variable before importing netket or jax, e.g.:
```python
import os
os.environ['NETKET_EXPERIMENTAL_SHARDING_CPU'] = '8'

import netket as nk
# ...
```
You should only use this to test things that they work, but not for anything serious. It has relatively bad performance compared to GPU-based sharding.


(jax_multi_process)=
### Sharding: Multiple nodes

To launch netket on a multi-node cluster usually all that is required is to add a call to `jax.distributed.initialize()` at the top of the main script, see the following examples.
These scripts can be conveniently launched with `srun` (on slurm clusters) or with job schedulers that support multi-process execution.
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


:::{note}
### Background:
_Jax_ internally uses the [grpc library](https://grpc.io) (launching a http server) for setup and book-keeping of the cluster and the [nvidia nccl library](https://developer.nvidia.com/nccl) for communication between gpus, and (experimentally) MPI or [gloo](https://github.com/facebookincubator/gloo) for communication between cpus.
:::

#### Sharding: Locally testing multi-process sharding

To test multi-process sharding locally on a computer the most reliable way is to launch multiple instances of your script with mpi, setup their communication channels with `mpi4py`, and then use the (very slow) ``gloo`` cpu communication library to make the multiple processes talk among them.

We use this to run tests on GitHub, for example, or to test scripts locally, but you should not use it for production.
To use this setup, you just need to install MPI on your computer and mpi4py in your python environment.

```python
import jax

# Use GLOO for CPU-to CPU communication. This is very slow.
# You could also set 'mpi' here, which has same performance as normal MPI
# but it's a pain to setup. see instructions below for MPITrampoline.
jax.config.update("jax_cpu_collectives_implementation", "gloo")
# This assumes you have mpi4py installed, but is very reliable.
print("initializing jax distributed...", flush=True)
jax.distributed.initialize(cluster_detection_method="mpi4py")
print("initialization succeded...", flush=True)

import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
# ...

if jax.process_index() == 0:
    print("only printed from rank 0", flush=True)
print(f"On process {jax.process_index()} I see the devices {jax.local_devices()} out of {jax.devices()}", flush=True)
```

and then you launch it with the command

```bash
mpirun -np 2 python yourscript.py
```

This will run netket with sharding mode, and will only use MPI to launch multiple copies of python
and to setup the communication among different ranks.

Do note that for CPUs this will be very slow. While if you have multiple GPUs, it will default to using
a single GPU per process.

:::{note}
See the [Sharding/multi_process.py](https://github.com/netket/netket/blob/master/Examples/Sharding/multi_process.py) example in the NetKet repository for a complete example.
:::

:::{warning}
### Common problems

Common issues we have found so far with this setup are:
 - On MacOs, this initialization is sometimes not compatible with OpenMPI>=5
 - If the 'hostname' does not resolve to 127.0.0.1, the call to `.initlaize` will deadlock. In those cases you just need to bind your hostname to `127.0.0.1`. This sometimes happen when using VPNs at some institutions.

About this latter point: if the script below does not work and deadlocks on your local computer right after printing ``initializing jax distributed`, it is often caused by the fact that your hostname is not resolvable.
To diagnose the issue, run the following command

```bash
ping $(hostname)
``` 
If it fails, then it means that your computer does not know how to talk to itself through its own hostname.
To fix this, you must edit the `/etc/host` file and add a line that looks like
```bash
echo "127.0.0.1      $(hostname)"
```
To edit the host file, you can for example do ``sudo nano \etc\host``
:::

#### MPITrampoline backend (very experimental)

MPITrampoline can be used in place of GLOO to make different CPU processes communicate, making sharding run as fast as MPI for multi-process CPU computations.
This works, but your code will only run as well as normal MPI code, and it's complex to install, so we don't particularly recomend it.

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
#### GRPC incompatibility with http proxy wildcards
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
