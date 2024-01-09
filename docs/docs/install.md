# Installation

Installing NetKet is very easy, but it has several complex and optional dependencies. In general, we suggest to create a virtual-environment for every project you work on.

## Installing NetKet

Netket requires `python>= 3.9` and can optionally benefit from a recent MPI install.
GPUs are supported on linux.

Before attempting the installation, you should update `pip` to a recent version (`>=20.3`) to avoid getting a broken install.
To install the basic version with no optional dependencies, run the following commands:

```bash
pip install --upgrade pip
pip install --upgrade netket
```

To query the installed `netket` version you can run the following command in your shell. 
If all went well, you should have at least version 3.3 installed. 
We recommend to always start a new project with the latest available version.

```bash
python -c "import netket; print(netket.__version__)"
```

```{admonition} Conda
:class: warning

If you are using a [`conda`](https://docs.conda.io/en/latest/) environment, please don't.
If you really want to use [`conda`](https://docs.conda.io/en/latest/) environments, see [this section](conda).
```

```{admonition} Install Errors?
:class: seealso

If you experience an installation error under `pip`, please make sure you have upgraded pip first and that you are not inside of a conda environment. If the install succeeds but you can't load `netket`, you most likely need to update the dependencies.
To get help, please open an issue pasting the output of `python -m netket.tools.info`.
```

## GPU support

If you want to run NetKet on a GPU, you must install a GPU-compatible {code}`jaxlib`, which is only supported on Linux and requires `CUDA>=11` and `cuDNN>=8.2`.
We advise you to look at the instructions on [jax repository](https://github.com/google/jax#pip-installation-gpu-cuda) because they change from time to time.
At the time of writing, installing a GPU version of jaxlib is as simple as running the following command, assuming you have very recent versions of `CUDA` and `cuDNN`.

```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Where the jaxlib version must correspond to the version of the existing CUDA installation you want to use. 
Refer to jax documentation to learn more about matching cuda versions with python wheels.


(install_mpi)=
## MPI

NetKet (due to Jax) only uses 1 or 2 CPU cores or 1 GPU by default at once unless you work on huge systems with mastodontic neural-networks. 
If you want to use all your CPU cores, multiple GPUs, or run your code among many computers you'll need to use MPI.
If you want to use MPI, make sure mpi is installed and can be used. 
To know if MPI is installed, try running the following command.

```
mpicc --showme:link
```

If the command fails, you need to install MPI using your favourite package manager. 
In general, for the love of yourself and in order to keep you sanity, *we recommend not to use `conda` together with MPI*.

 - On Mac, we recommend to use homebrew:

```
brew install openmpi
```

 - On Linux, you can install it using your package manager:

```bash
# fedora
sudo dnf install mpich
# ubuntu/debian
sudo apt-get install mpich
```

You can install the dependencies necessary to run with MPI with the following command:

```bash
pip install --upgrade
pip install --upgrade "netket[mpi]"
```

Subsequently, NetKet will exploit MPI-level parallelism for the Monte-Carlo sampling.
See {ref}`this block <warn-mpi-sampling>` to understand how NetKet behaves under MPI.

(conda)=
## Conda

Conda is a great package manager as long as it works. 
But when it does not, it's a pain.

To install NetKet using conda, simply run

```bash
conda install -c conda-forge netket
```

This will also install the conda MPI compilers and the MPI-related dependencies. 
This often creates problems if you also have a system MPI. 
Moreover, you should never use conda's MPI on a supercomputing cluster.

In general, we advise against using conda or conda environments to install NetKet unless someone is pointing a gun at you.
If you don't want to die from that bullet, but would rather loose your mental sanity fighting conda, do expect weird setup errors.


## Introduction

Netket is a numerical framework written in Python to simulate many-body quantum systems using
variational methods. In general, netket allows the user to parametrize quantum states using
arbitrary functions, be it simple mean-field Ansätze, Jastrow, MPS Ansätze or convolutional
neural networks.
Those states can be sampled efficiently in order to estimate observables or other quantities.
Stochastic optimisation of the energy or a time-evolution are implemented on top of those samplers.

Netket tries to follow the [functional programming](https://en.wikipedia.org/wiki/Functional_programming) paradigm,
and is built around [jax](https://en.wikipedia.org/wiki/Functional_programming). While it is possible
to run the examples without knowledge of [jax], we strongly recommend getting familiar with it if you
wish to extend netket.

This documentation is divided into several modules, each explaining in-depth how a sub-module of netket works.
You can select a module from the list on the left, or you can read the following example which contains links
to all relevant parts of the documentation.

## Jax/Flax extensions

NetKet v3 API is centered around [flax](https://flax.readthedocs.io), a JAX-based library providing components to define and use neural network models.
If you want to define more complex custom models, you should read Flax documentation on how to define a Linen module.
If you wish, you can also use [haiku](https://github.com/deepmind/dm-haiku).
[`netket.optimizer`](netket_optimizer_api) is a re-export of some optimizers from [optax](https://optax.readthedocs.io) together with some additional objects.

Lastly, in [`netket.jax`](netket_jax_api) there are a few functions, notably `jax.grad` and `jax.vjp` adapted to work with arbitrary real or complex functions, and/or with MPI.


## Commented Example

```python
import netket as nk
import numpy as np
```

The first thing to do is import NetKet. We usually shorten it to `nk`.

```python
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
```

Then, one must define the system to be studied.
For a discrete system, the first thing to do is usually defining the underlying lattice structure of the model.
Several types of lattices and, more generally, graphs allowing arbitrary connections between sites are defined in the [Graph submodule](netket_graph_api).

In the example above we chose a 1-Dimensional chain with 20 sites and periodic boundary conditions.

```python
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
```

Then, one must define the hilbert space and the hamiltonian. 
Common options for the Hilbert space are {class}`~nk.hilbert.Spin`,  {class}`~nk.hilbert.Fock` or {class}`~nk.hilbert.Qubit`, but it is also possible to define your own.
Those classes are contained in the [Hilbert submodule](hilbert.md).

The [operator sub-module](netket_operator_api) contains several pre-built hamiltonian, such as {class}`~nk.operator.Ising` and {class}`~nk.operator.Bose-Hubbard`, but you can also build the operators yourself by summing all the local terms. 
See the [Operators](operator.md) documentation for more information.

```python
ma = nk.models.RBM(alpha=1, param_dtype=float)

sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
```

Then, one must chose the model to use as a Neural Quantum State. Netket provides
a few pre-built models in the [Models sub-module](netket_models_api).
Netket models are simply [Flax] modules: check out the [defining custom models section](custom-models) for more information on how to define or use custom models.
We specify {code}`param_dtype=float` (which is the default, but we want to show
it to you) which means that weights will be stored as double-precision.
We advise you that Jax (and therefore netket) does not follow completely the standard NumPy promotion rules, instead treating {code}`float` as a weak double-precision type
which can \_loose\_ precision in some cases.
This can happen if you mix single and double precision in your models and the sampler and
is described in [Jax:Type promotion semantics](https://jax.readthedocs.io/en/latest/type_promotion.html).

Hilbert space samplers are defined in the [Sampler submodule](sampler.ipynb).
In general you must provide the constructor of the hilbert space to be sampled and some options.
In this case we ask for 16 markov chains.
The default behaviour for samplers is to output states with double precision, but
this can be configured by specifying the {code}`dtype` argument when constructing the
sampler.
Samples don't need double precision at all, so it makes sense to use the lower
precision, but you have to be careful with the dtype of your model in order
not to reduce the precision.

```python
# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)
```

You can then chose an optimizer from the [optimizer](netket_optimizer_api) submodule. You can also use an arbitrary flax optimiser, or define your own.

```python
# Variational monte carlo driver
gs = nk.VMC(ha, op, sa, ma, n_samples=1000, n_discard_per_chain=100)

gs.run(n_iter=300, out=None)
```

Once you have all the pieces together, you can construct a variational monte
carlo optimisation driver by passing the constructor the hamiltonian and the
optimizer (which must always be the first two arguments), and then the
sampler, machine and various options.

Once that is done, you can run the simulation by calling the {meth}`~nk.driver.VMC.run` method in the driver, specifying the output loggers and the number of iterations in
the optimisation.
