
# <img src="http://www.netket.org/img/logo_simple.jpg" width="400"> <img src="http://www.netket.org/img/logo_simple.jpg" width="400">

[![Release](https://img.shields.io/github/release/netket/netket.svg)](https://github.com/netket/netket/releases)
[![Build Status](https://travis-ci.org/netket/netket.svg?branch=master)](https://travis-ci.org/netket/netket)
[![CodeFactor](https://www.codefactor.io/repository/github/netket/netket/badge)](https://www.codefactor.io/repository/github/netket/netket)
[![GitHub Issues](https://img.shields.io/github/issues/netket/netket.svg)](http://github.com/netket/netket/issues)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jamesETsmith/netket/mybinder)


# __NetKet__

NetKet is an open-source project delivering cutting-edge methods for the study
of many-body quantum systems with artificial neural networks and machine learning techniques.


## Major Features

* Graphs
  * Built-in Graphs
    * Hypercube
  * Custom Graphs
    * Any Graph With Given Adjacency Matrix [from input file]
    * Any Graph With Given Edges [from input file]
  * Symmetries
    * Automorphisms: pre-computed in built-in graphs, available through iGraph for custom graphs

* Hamiltonians
  * Built-in Hamiltonians
    * Transverse-field Ising
    * Heisenberg
    * Bose-Hubbard
  * Custom Hamiltonians
    * General k-local Hamiltonians defined on Graphs
    * Any k-local Hamiltonian [from input file]

 * Ground State Solvers  
   * Stochastic Learning Methods
     * Gradient Descent
     * Stochastic Reconfiguration Method
       * Direct Solver
       * Iterative Solver for Large Number of Parameters  
   * Exact Diagonalization
     * Full Solver
     * Lanczos Solver
     * Imaginary-Time Dynamics

 * Optimizers
    * Stochastic Gradient Descent
    * AdaMax, AdaDelta, AdaGrad, AMSGrad
    * RMSProp
    * Momentum
    * Gradient Clipping

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
  * Jastrow wavefunction
    * Standard
    * With Permutation Symmetry Using Graph Isomorphisms
  * Custom Machines
    * Any Machine Satisfying Prototypes in `AbstractMachine` [extending C++ code]

* Observables
  * Custom Observables
    * Any k-local Operator [from input file]

* Sampling
  * Local Metropolis Moves
    * Local Hilbert Space Sampling
    * Parallel Tempering Versions
  * Hamiltonian Moves
    * Automatic Moves with Hamiltonian Symmetry
    * Parallel Tempering Versions
  * Custom Sampling
    * Any k-local Stochastic Operator can be used to do Metropolis Sampling

* Statistics
  * Automatic Estimate of Correlation Times

* I/O
  * Python/JSON Interface   

## Installation and Usage

Please visit our [homepage](https://www.netket.org) for further information.

## License

[Apache License 2.0](https://github.com/netket/netket/blob/master/LICENSE)
