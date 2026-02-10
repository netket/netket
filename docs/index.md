NetKet: The Machine Learning Toolbox for Quantum Physics
===========================================

NetKet is a Python library for using machine learning methods to study many-body quantum systems. It provides efficient and flexible building blocks for writing novel algorithms as well as simple, easy-to-use implementations of established algorithms.

NetKet is built on top of [JAX](https://jax.readthedocs.io), a framework for differentiable programming that works on CPUs, GPUs and TPUs. Neural Network architectures can be specified using any JAX-based framework such as [Flax](https://flax.readthedocs.io).

::::{grid} 4
:class-container: color-cards

:::{grid-item-card} 💻 Installation
:columns: 12 6 6 3
:link: install
:link-type: doc
:class-card: installation

Get NetKet up and running on your system
:::

:::{grid-item-card} 🚀 Getting started
:columns: 12 6 6 3
:link: tutorials/gs-ising
:link-type: doc
:class-card: getting-started

Learn NetKet with hands-on tutorials
:::

:::{grid-item-card} 📚 User guides
:columns: 12 6 6 3
:link: user-guides/index-modules
:link-type: doc
:class-card: user-guides

In-depth guides for NetKet components
:::

:::{grid-item-card} 🔬 Examples
:columns: 12 6 6 3
:link: https://github.com/netket/netket/tree/master/Examples
:link-type: url
:class-card: examples

Short runnable scripts showcasing features
:::
::::

## Supporting and Citing

The software in this ecosystem was developed as part of academic research.
If you would like to help support it, please star the repository as such metrics may help us secure funding in the future.
If you use NetKet software as part of your research, teaching, or other activities, we would be grateful if you could cite our work.

Guidelines on citation are provided in the [Citing](https://www.netket.org/cite) section of our website.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

install
First Tutorial <tutorials/gs-ising>
parallel
distributed-computing
cite
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: User guides

philosophy
tutorials/index
user-guides/index-modules
advanced/index
vmc-from-scratch/index
developer-guides/index
user-guides/configurations
sharp-bits

```


```{toctree}
:hidden:
:maxdepth: 2
:caption: API documentation

api/api-stability
api/api
api/api-experimental
changelog
```
