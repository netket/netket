// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_ABSTRACTHAMILTONIAN_HPP
#define NETKET_ABSTRACTHAMILTONIAN_HPP

#include <Eigen/Dense>
#include <complex>
#include <vector>

namespace netket {

/**
      Abstract class for Hamiltonians.
      This class prototypes the methods needed
      by a class satisfying the Hamiltonian concept.
      Users interested in implementing new hamiltonian should derive they own
      class from this class.
*/
class AbstractHamiltonian {
public:
  /**
  Member function finding the connected elements of the Hamiltonian.
  Starting from a given visible state v, it finds all other visible states v'
  such that the hamiltonian matrix element H(v,v') is different from zero.
  In general there will be several different connected visible units satisfying
  this condition, and they are denoted here v'(k), for k=0,1...N_connected.
  @param v a constant reference to the visible configuration.
  @param mel(k) is modified to contain matrix elements H(v,v'(k)).
  @param connector(k) for each k contains a list of sites that should be changed
  to obtain v'(k) starting from v.
  @param newconfs(k) is a vector containing the new values of the visible units
  on the affected sites, such that: v'(k,connectors(k,j))=newconfs(k,j). For the
  other sites v'(k)=v, i.e. they are equal to the starting visible
  configuration.
  */
  virtual void FindConn(const Eigen::VectorXd &v,
                        std::vector<std::complex<double>> &mel,
                        std::vector<std::vector<int>> &connectors,
                        std::vector<std::vector<double>> &newconfs) = 0;

  /**
  Member function returning the hilbert space associated with this Hamiltonian.
  @return Hilbert space specifier for this Hamiltonian
  */
  virtual const Hilbert &GetHilbert() const = 0;

  virtual ~AbstractHamiltonian() {}
};
} // namespace netket

#endif
