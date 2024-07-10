# 🔪 The Sharp Bits 🔪

Read ahead for some pitfalls, counter-intuitive behavior, and sharp edges that we had to introduce in order to make this work.

(gpus)=
## Using GPUs

Jax supports GPUs, so your calculations should run fine on GPU, however there are a few gotchas:

- GPUs have a much higher overhead, therefore you will see very bad performance at small system size (typically below 40 spins)
- Not all Metropolis Transition Rules work on GPUs. To go around that, those rules have been rewritten in numpy in order to run on the cpu, therefore you might need to use {ref}`netket.sampler.MetropolisSamplerNumpy` instead of {ref}`netket.sampler.MetropolisSampler`.

Eventually we would like the selection to be automatic, but this has not yet been implemented.

Please open tickets if you find issues!


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
