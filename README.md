
# <img src="http://www.netket.org/img/logo_simple.jpg" width="400"> <img src="http://www.netket.org/img/logo_simple.jpg" width="400">

[![Build Status](https://travis-ci.org/netket/netket.svg?branch=master)](https://travis-ci.org/netket/netket)
# __NetKet__

NetKet is an open-source project delivering cutting-edge methods for the study
of many-body quantum systems with artificial neural networks and machine learning techniques.


## Major Features

* Graphs
  * Built-in Graphs
    * Hypercube
  * Custom Graphs
    * Any Graph With Given Adjacency Matrix [from input file]

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
    * With Permutation Symmetry
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

Please visit our [homepage](http://www.netket.org) for further information.

## License

[Apache License 2.0](https://github.com/netket/netket/blob/master/LICENSE)
