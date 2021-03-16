<div align="center">
<img src="https://www.netket.org/_static/logo_simple.jpg" alt="logo" width="400"></img>
</div>

# __NetKet__

[![Release](https://img.shields.io/github/release/netket/netket.svg)](https://github.com/netket/netket/releases)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/netket/badges/installer/conda.svg)](https://conda.anaconda.org/conda-forge)
[![Build Status](https://travis-ci.org/netket/netket.svg?branch=master)](https://travis-ci.org/netket/netket)
[![GitHub Issues](https://img.shields.io/github/issues/netket/netket.svg)](http://github.com/netket/netket/issues)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A1904.00031-B31B1B.svg)](https://arxiv.org/abs/1904.00031)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/netket/netket/v.2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

NetKet is an open-source project delivering cutting-edge methods for the study
of many-body quantum systems with artificial neural networks and machine learning techniques.
It is a Python library built on C++ primitives.

- **Homepage:** <https://netket.org>
- **Citing:** <https://www.netket.org/citing>
- **Documentation:** <https://netket.org/documentation>
- **Tutorials:** <https://www.netket.org/tutorials>
- **Examples:** <https://github.com/netket/netket/tree/master/Examples>
- **Source code:** <https://github.com/netket/netket>

## Installation and Usage
Netket supports MacOS and Linux. The reccomended way to install it in a non-conda python environment is: 
```
pip install netket[mpi]
``` 
The `[mpi]` after netket will install mpi related dependencies of netket. 
We reccomend to install netket with all it's extra dependencies, which are documented below. 
However, if you do not have a working MPI compiler in your PATH this installation will most likely fail because
it will attempt to install `mpi4py`, which enables MPI support in netket.
If you are only starting to discover netket and won't be running extended simulations, you can forego MPI by 
installing netket with the command
```
pip install netket 
```

Netket is also available on conda-forge. To install netket in a conda-environment you can use:
```
conda install conda-forge::netket
```
The conda library is linked to anaconda's `mpi4py`, therefore we do not reccomend to use this installation
method on computer clusters with a custom MPI distribution. 
We don't reccomend to install from conda as the jaxlib there is not very performant.

### Extra dependencies
When installing netket with pip, you can pass the following extra variants as square brakets. You can install several of them by separating them with a comma.
 - '[dev]': installs development-related dependencies such as black, pytest and testing dependencies
 - '[mpi]': Installs `mpi4py` to enable multi-process parallelism. Requires a working MPI compiler in your path
 - '[all]': Installs `mpi`, and `dev`.

### MPI Support
Depending on the library you use to define your machines, distributed computing through MPI might
or might not be supported. Please see below:
  - **netket** : distributed computing through MPI support can be enabled by installing the package `mpi4py` through pip or conda.
  - **jax**    : distributed computing through MPI is supported natively only if you don't use Stochastic Reconfiguration (SR). If you need SR, you must install [mpi4jax](https://github.com/PhilipVinc/mpi4jax). Please note that we advise to install mpi4jax  with the same tool (conda or pip) with which you installed netket.
  - **pytorch** : distributed computing through MPI is enabled if the package `mpi4py` is isntalled. Stochastic Reconfiguration (SR) cannot be used when MPI is enabled. 

To check whever MPI support is enabled, check the flags 
```python
# For standard MPI support
>>> netket.utils.mpi_available
True

# For faster MPI support with jax and to enable SR + MPI with Jax machines
>>> netket.utils.mpi4jax_available
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

* Machines
  * Restricted Boltzmann Machines
    * Standard
    * For Custom Local Hilbert Spaces
    * With Permutation Symmetry Using Graph Isomorphisms
  * Feed-Forward Networks
    * For Custom Local Hilbert Spaces
    * Fully connected layer
    * Convnet layer for arbitrary underlying graph
    * Any Layer Satisfying Prototypes in `AbstractLayer` [extending C++ code]
  * Jastrow States
    * Standard
    * With Permutation Symmetry Using Graph Isomorphisms
  * Matrix Product States
    * MPS
    * Periodic MPS  
  * Custom Machines
    * Any Machine Satisfying Prototypes in `AbstractMachine` [extending C++ code]

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
  * Python Library
  * JSON output  

## License

[Apache License 2.0](https://github.com/netket/netket/blob/master/LICENSE)
