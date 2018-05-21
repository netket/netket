
# <img src="http://www.netket.org/img/logo_simple.jpg" width="400"> <img src="http://www.netket.org/img/logo_simple.jpg" width="400">

[![Release](https://img.shields.io/github/release/netket/netket.svg)](https://github.com/netket/netket/releases)
[![Build Status](https://travis-ci.org/netket/netket.svg?branch=master)](https://travis-ci.org/netket/netket)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/acfc9fcbedd54b77a2d45351f4518728)](https://www.codacy.com/app/gcarleo/netket?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=netket/netket&amp;utm_campaign=Badge_Grade)
[![GitHub Issues](https://img.shields.io/github/issues/netket/netket.svg)](http://github.com/netket/netket/issues)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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

* Hamiltonians
  * Built-in Hamiltonians
    * Transverse-field Ising
    * Heisenberg
    * Bose-Hubbard
  * Custom Hamiltonians
    * Any k-local Hamiltonian [from input file]

* Learning
  * Steppers
    * Stochastic Gradient Descent
    * AdaMax
  * Ground-state Learning
    * Gradient Descent
    * Stochastic Reconfiguration Method
      * Direct Solver
      * Iterative Solver for Large Number of Parameters  

* Machines
  * Restricted Boltzmann Machines
    * Standard
    * For Custom Local Hilbert Spaces
    * With Permutation Symmetry Using Graph Isomorphisms
  * Custom Machines
    * Any Machine Satisfying Prototype of Abstract Machine [extending C++ code]

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

* Statistics
  * Automatic Estimate of Correlation Times

* I/O
  * Python/JSON Interface   

## Installation and Usage

Please visit our [homepage](https://www.netket.org) for further information.

## License

[Apache License 2.0](https://github.com/netket/netket/blob/master/LICENSE)
