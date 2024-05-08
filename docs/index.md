# NetKet: The Machine Learning Toolbox for Quantum Physics

NetKet is a suite for using Machine-Learning methods to numerically study many-body quantum systems and available for use in Python.
The purpose of this package is to supply efficient and flexible building blocks to write novel algorithms as well as to provide simple, easy to use implementations of established algorithms.

NetKet is built on top of [Jax], a framework for Differentiable Programming which works on CPUs, GPUs and TPUs. Neural Network architectures can be specified using any Jax-based framework such as [Flax].

Some of the tasks that NetKet can be used for are:

 - Variational Ground State search
 - Variational dynamics
 - Variational tomography
 - Bosonic and Fermionic models

NetKet includes those interesting :

 - Support for arbitrary periodic Lattices
   - Automatic generation of symmetry groups and character tables
 - Implementation of Autoregressive Neural Networks
 - Implementation of symmetry-invariant and -equivariant networks.

## Getting Started and Tutorials

The best way to learn how to use NetKet is to follow along the tutorials listed in the tutorial section on the left navigation bar.
The first few tutorial, [Ising model: ground-state search](tutorials/gs-ising), gives a very broad overview of the workflow when working with NetKet, and how to define a Neural-Network quantum state.
Then, you can move on to more advanced tutorials.

All notebooks can be launched on Google Colab (an online python environment) by clicking on the small rocket icon on the top bar.
We suggest you to read them while executing them on Colab to experiment.

If you have questions, don't hesitate to start a discussion on the [Github forum](https://github.com/netket/netket/discussions).


## Supporting and Citing

The software in this ecosystem was developed as part of academic research.
If you would like to help support it, please star the repository as such metrics may help us secure funding in the future.
If you use NetKet software as part of your research, teaching, or other activities, we would be grateful if you could cite our work.

Guidelines on citation are provided in the [Citation](https://www.netket.org/citation) section of our website.

## Table of Contents

```{toctree}
:caption: Getting Started
:maxdepth: 1

docs/install
docs/parallelization
docs/sharp-bits
```   

```{toctree}
:caption: Tutorials
:maxdepth: 2

tutorials/gs-ising
tutorials/gs-continuous-space
tutorials/gs-heisenberg
tutorials/gs-j1j2
tutorials/gs-matrix-models
tutorials/gs-gcnn-honeycomb
tutorials/lattice-fermions
tutorials/vmc-from-scratch
```   

```{toctree}
:caption: Reference Documentation
:maxdepth: 2

docs/hilbert
docs/operator
docs/sampler
docs/varstate
docs/sr
docs/drivers
docs/superop
```

```{toctree}
:maxdepth: 2
:caption: Extending NetKet
:hidden:

advanced/custom_models
advanced/custom_expect
advanced/custom_preconditioners
advanced/custom_operators
```

```{toctree}
:maxdepth: 3
:caption: API documentation
:hidden:

api/api-stability
docs/changelog
docs/configurations
api/api
api/api-experimental
```

```{toctree}
:caption: Developer Documentation
:maxdepth: 2

docs/contributing
docs/writing-tests
```


[Jax]: https://jax.readthedocs.com "Jax"
[Flax]: https://flax.readthedocs.com "Flax"
[Optax]: https://optax.readthedocs.com "Optax"
