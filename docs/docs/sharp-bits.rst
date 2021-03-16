🔪 The Sharp Bits 🔪
====================

Read ahead for some pitfalls, counter-intuitive behavior, and sharp edges that we had to introduce in order to make this work.

.. _parallelization:

Parallelization
---------------

Netket computations run mostly via Jax's XLA. 
Compared to NetKet 2, this means that we can automatically benefit from multiple CPUs without having to use MPI.
This is because mathematical operations such as matrix multiplications and overs will be split into sub-chunks and distributed across different cpus. 
This behaviour is trigered only for matrices/vectors above a certain size, and will not perform particularly good for small matrices or if you have many cpu cores.
To disable this behaviour, refer to `Jax#743 <https://github.com/google/jax/issues/743>`_, which mainly suggest defining the two env variables: 

.. code:: bash

	export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

Usually we have noticed that the best performance is achieved by combining both BLAS parallelism and MPI, for example by guaranteeing between 2-4 (depending on your problem size) cpus to every MPI thread.



.. _running_on_cpu:

Running on CPU when GPUs are present
--------------------------------------

If you have the CUDA version of jaxlib installed, then computations will, by default, run on the GPU.
For small systems this will be very inefficient. To check if this is the case, run the following code:

.. code:: python

	import jax
	print(jax.devices())

If the output is :code:`[CpuDevice(id=0)]`, then computations will run by default on the CPU, if instead you see
something like :code:`[GpuDevice(id=0)]` computations will run on the GPU.

To force Jax/XLA to run comutations on the CPU, set the environment variable 

.. code:: bash

	export JAX_PLATFORM_NAME="cpu"



.. _gpus:

Using GPUs
----------

Jax supports GPUs, so your calculations should run fine on GPU, however there are a few gotchas:

* GPUs have a much higher overhead, therefore you will see very bad performance at small system size (typically below 40 spins)

* Not all Metropolis Transition Rules work on GPUs. To go around that, those rules have been rewritten in numpy in order to run on the cpu, therefore you might need to use :ref:`netket.sampler.MetropolisSamplerNumpy` instead of :ref:`netket.sampler.MetropolisSampler`.

Eventually we would like the selection to be automatic, but this has not yet been implemented. 

Please open tickets if you find issues!