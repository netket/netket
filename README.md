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
You can install on osx or linux with either
 - *pip*   : `pip install netket`
 - *conda* : `conda install conda-forge::netket`

Conda by default ships pre-built binaries for recent versions of python.
The default blas library is openblas, but mkl can be enforced.

To learn more, check out the website or the examples.

Since version 3, in addition to the built-in machines, you can also use [Jax](https://github.com/google/jax) and [PyTorch](https://pytorch.org) to define your custom neural networks.
Those are not hard dependencies of netket, therefore they must be installed separately.
To avoid potential bugs, we suggest to install those libraries using the same tool that you
used for netket (pip or conda).

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
