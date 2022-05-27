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
