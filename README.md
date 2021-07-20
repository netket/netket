<div align="center">
<img src="https://www.netket.org/_static/logo_simple.jpg" alt="logo" width="400"></img>
</div>

# __NetKet__

[![Release](https://img.shields.io/github/release/netket/netket.svg)](https://github.com/netket/netket/releases)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/netket/badges/version.svg)](https://anaconda.org/conda-forge/netket)
[![Paper](https://img.shields.io/badge/paper-SoftwareX%2010%2C%20100311%20(2019)-B31B1B)](https://doi.org/10.1016/j.softx.2019.100311)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/netket/netket/branch/master/graph/badge.svg?token=gzcOlpO5lB)](https://codecov.io/gh/netket/netket)

NetKet is an open-source project delivering cutting-edge methods for the study
of many-body quantum systems with artificial neural networks and machine learning techniques.
It is a Python library built on [JAX](https://github.com/google/jax).

- **Homepage:** <https://www.netket.org>
- **Citing:** <https://www.netket.org/citing>
- **Documentation:** <https://www.netket.org/documentation>
- **Tutorials:** <https://www.netket.org/tutorials>
- **Examples:** <https://github.com/netket/netket/tree/master/Examples>
- **Source code:** <https://github.com/netket/netket>

## Installation and Usage

Netket supports MacOS and Linux. We reccomend to install NetKet using `pip`
For instructions on how to install the latest stable/beta release of NetKet see the [Getting Started](https://www.netket.org/website/get_started.html) section of our website.

If you wish to install the current development version of NetKet, which is the master branch of this GitHub repository, together with the additional
dependencies, you can run the following command:

```
pip install 'git+https://github.com/netket/netket.git#egg=netket[all]'
``` 

You can also install the MPI-related dependencies by using `[dev,mpi]` between the square brackets.
We recommend to install NetKet with all it's extra dependencies, which are documented below. 
However, if you do not have a working MPI compiler in your PATH this installation will most likely fail because
it will attempt to install `mpi4py`, which enables MPI support in netket.

The latest release of Netket is not currently available on conda-forge. 
However, you can still install NetKet with pip inside conda environments.

### Extra dependencies
When installing netket with pip, you can pass the following extra variants as square brakets. You can install several of them by separating them with a comma.
 - '[dev]': installs development-related dependencies such as black, pytest and testing dependencies
 - '[mpi]': Installs `mpi4py` to enable multi-process parallelism. Requires a working MPI compiler in your path
 - '[tensorboard]': Installs `tensorboardx` to enable logging to tensorboard.
 - '[all]': Installs all extra dependencies

### MPI Support
To enable MPI support you must install [mpi4jax](https://github.com/PhilipVinc/mpi4jax). Please note that we advise to install mpi4jax  with the same tool (conda or pip) with which you install it's dependency `mpi4py`.

To check whever MPI support is enabled, check the flags 
```python
>>> import netket
>>> netket.utils.mpi.available
True

```

## Major Features

* Graphs
  * Built-in Graphs
    * Hypercube
    * General Lattice with arbitrary number of atoms per unit cell
  * Custom Graphs
    * Any Graph With Given Adjacency Matrix
    * Any Graph With Given Edges
  * Symmetries
    * Automorphisms: pre-computed in built-in graphs, available through iGraph for custom graphs

* Quantum Operators
  * Built-in Hamiltonians
    * Transverse-field Ising
    * Heisenberg
    * Bose-Hubbard
  * Custom Operators
    * Any k-local Hamiltonian
    * General k-local Operator defined on Graphs

* Variational Monte Carlo   
  * Stochastic Learning Methods for Ground-State Problems
    * Gradient Descent
    * Stochastic Reconfiguration Method
      * Direct Solver
      * Iterative Solver for Large Number of Parameters  

* Exact Diagonalization
  * Full Solver
  * Lanczos Solver
  * Imaginary-Time Dynamics

* Supervised Learning
  * Supervised overlap optimization from given data

* Neural-Network Quantum State Tomography
  * Using arbitrary k-local measurement basis       

* Optimizers
  * Stochastic Gradient Descent
  * AdaMax, AdaDelta, AdaGrad, AMSGrad
  * RMSProp
  * Momentum

* Models
  * Restricted Boltzmann Machines
    * Standard
    * For Custom Local Hilbert Spaces
    * With Permutation Symmetry Using Graph Isomorphisms
  * Feed-Forward Networks
    * For Custom Local Hilbert Spaces
  * Jastrow States
    * Standard
    * With Permutation Symmetry Using Graph Isomorphisms
  * Matrix Product States
    * MPS
    * Periodic MPS
  * Custom Models

* Observables
  * Custom Observables
    * Any k-local Operator

* Sampling
  * Local Metropolis Moves
    * Local Hilbert Space Sampling
  * Hamiltonian Moves
    * Automatic Moves with Hamiltonian Symmetry
  * Custom Sampling
    * Any k-local Stochastic Operator can be used to do Metropolis Sampling
  * Exact Sampler for small systems  

* Statistics
  * Automatic Estimate of Correlation Times

* Interface
  * Python module
  * JSON output

## License

[Apache License 2.0](https://github.com/netket/netket/blob/master/LICENSE)
