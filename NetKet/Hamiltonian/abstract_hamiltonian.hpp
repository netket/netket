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

#include "Hilbert/hilbert.hpp"

namespace netket {

/**
 * This struct represents a non-zero matrix-element H(v,v') of an operator for a
 * given visible state v.
 */
struct MatrixElement {
  std::complex<double> weight;    /// The matrix element H(v,v')
  ConfigurationUpdateRef update;  /// The update neccessary to obtain v' from v
};

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
                        std::vector<std::vector<double>> &newconfs) const = 0;

  using ConnCallback = std::function<void(MatrixElement)>;
  /**
   * Iterates over all states reachable from a given visible configuration v,
   * i.e., all states v' such that H(v,v') is non-zero.
   * @param v The visible configuration.
   * @param callback Function void callback(MatrixElement mel) which will be
   * called once for each reachable configuration v'. The parameter mel contains
   * the value H(v,v') and the information to obtain v' from v. Note that the
   * member mel.update has reference character and can only be savely used
   * inside the callback. It will become invalid once callback returns.
   */
  virtual void ForEachConn(const Eigen::VectorXd &v,
                           ConnCallback callback) const;

  /**
  Member function returning the hilbert space associated with this Hamiltonian.
  @return Hilbert space specifier for this Hamiltonian
  */
  virtual const Hilbert &GetHilbert() const = 0;

  virtual ~AbstractHamiltonian() {}
};

void AbstractHamiltonian::ForEachConn(const Eigen::VectorXd &v,
                                      ConnCallback callback) const {
  std::vector<std::complex<double>> weights;
  std::vector<std::vector<int>> connectors;
  std::vector<std::vector<double>> newconfs;

  FindConn(v, weights, connectors, newconfs);

  for (size_t k = 0; k < connectors.size(); k++) {
    const ConfigurationUpdateRef update{connectors[k], newconfs[k]};
    const MatrixElement mel{weights[k], update};
    callback(mel);
  }
}

}  // namespace netket

#endif
