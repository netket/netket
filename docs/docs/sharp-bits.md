# 🔪 The Sharp Bits 🔪

Read ahead for some pitfalls, counter-intuitive behavior, and sharp edges that we had to introduce in order to make this work.


(parallelization)=
## Parallelization

Netket computations run mostly via Jax's XLA.
Compared to NetKet 2, this means that we can automatically benefit from multiple CPUs without having to use MPI.
This is because mathematical operations such as matrix multiplications and overs will be split into sub-chunks and distributed across different cpus.
This behaviour is triggered only for matrices/vectors above a certain size, and will not perform particularly good for small matrices or if you have many cpu cores.
To disable this behaviour, refer to [Jax#743](https://github.com/google/jax/issues/743), which mainly suggest defining the two env variables:

```bash
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
```

Usually we have noticed that the best performance is achieved by combining both BLAS parallelism and MPI, for example by guaranteeing between 2-4 (depending on your problem size) cpus to every MPI thread.

Note that when using {code}`netket` it is crucial to run Python with the same implementation and version of MPI that the {code}`mpi4py` module is compiled against.
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


(running-on-cpu)=
## Running on CPU when GPUs are present

If you have the CUDA version of jaxlib installed, then computations will, by default, run on the GPU.
For small systems this will be very inefficient. To check if this is the case, run the following code:

```python
import jax
print(jax.devices())
```

If the output is {code}`[CpuDevice(id=0)]`, then computations will run by default on the CPU, if instead you see
something like {code}`[GpuDevice(id=0)]` computations will run on the GPU.

To force Jax/XLA to run computations on the CPU, set the environment variable

```bash
export JAX_PLATFORM_NAME="cpu"
```

(gpus)=

## Using GPUs

Jax supports GPUs, so your calculations should run fine on GPU, however there are a few gotchas:

- GPUs have a much higher overhead, therefore you will see very bad performance at small system size (typically below 40 spins)
- Not all Metropolis Transition Rules work on GPUs. To go around that, those rules have been rewritten in numpy in order to run on the cpu, therefore you might need to use {ref}`netket.sampler.MetropolisSamplerNumpy` instead of {ref}`netket.sampler.MetropolisSampler`.

Eventually we would like the selection to be automatic, but this has not yet been implemented.

Please open tickets if you find issues!

(jax_multi_process)=
## Running on multi-gpu clusters
Historically the main way to run NetKet in parallel has been to use MPI. However, with jax adding shared arrays and collective operations on multiple devices/nodes (see [here](https://jax.readthedocs.io/en/latest/jax_array_migration.html#jax-array-migration) and [here](https://jax.readthedocs.io/en/latest/multi_process.html)) we adapted NetKet to support it.
To run on a single node with multiple gpus all that is necessary is to set the flag `NETKET_EXPERIMENTAL_SHARDING=1`.

:::{warning}
This feature is still experimental and not everything may work perfectly right out of the box.
Any feedback, be it positive or negative, would be greatly appreciated.
:::


(jax_multi_process_setup)=
### Multi-node setup with jax.distributed
To launch netket on a multi-node gpu cluster usually all that is required is to add a call to `jax.distributed.initialize()` at the top of the main script, e.g. as follows:

```python
import jax
jax.distributed.initialize()

import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = 1

import netket as nk
# ...
```

Then it can be conveniently launched with `srun` (on slurm clusters) or `mpirun`.
For more details and manual setups we refer to the [jax documentation](https://jax.readthedocs.io/en/latest/multi_process.html).

Note that jax internally uses the [grpc library](https://grpc.io) (launching a http server) for setup and book-keeping of the cluster and the [nvidia nccl library](https://developer.nvidia.com/nccl) for communication between gpus. Thus it is required that `libnccl2` and `libnccl2-dev` are installed in addition to cuda.
Even if launched with mpirun, mpi is not actually used for communication, but the environment variables set by it are instead picked up by `jax.distributed.initialize`.

If you run into communication errors you might want to set the environment variable `NCCL_DEBUG=INFO` for detailed error messages.

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

(multi_device)=
### Multiple devices per process
In our testing it is best to use 1 process per gpu on the cluster.

Nevertheless, if you want to use multiple gpus per process you can force jax to do so by setting `local_device_ids`, e.g. extracting it from `CUDA_VISIBLE_DEVICES` as follows:

```python
import os
import jax
ldi = list(map(int, os.environ.get('CUDA_VISIBLE_DEVICES').split(',')))
jax.distributed.initialize(local_device_ids=ldi)
```

(nan)=
## NaNs in training and loss of precision

If you find NaNs while training, especially if you are using your own model, there might be a few reasons:

- It might simply be a precision issue, as you might be using single precision ({code}`np.float32`, {code}`np.complex64`) instead of double precision ({code}`np.float64`, {code}`np.complex128`). Be careful that if you use {code}`float` and {code}`complex` as dtype, they will not always behave as you expect!
  They are known as [weak dtypes](https://jax.readthedocs.io/en/latest/type_promotion.html?highlight=type-promotion), and when multiplied by a single-precision number they will be converted to single precision.
  This issue might manifest especially when using Flax, which respects type promotion, as opposed to `jax.example_libraries.stax`, which does not.
- Check the initial parameters. In the NetKet 2 models were always initialized with weights normally distributed.
  In Netket 3, `netket.nn` layers use the same default (normal distribution with standard deviation 0.01) but
  if you use general flax layers they might use different initializers.
  different initialisation distributions have particularly strong effects when working with complex-valued models.
  A good way to enforce the same distribution across all your weights, similar to NetKet 2 behaviour, is to use {py:meth}`~netket.vqs.VariationalState.init_parameters`.
