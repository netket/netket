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

#ifndef NETKET_ISING1D_HPP
#define NETKET_ISING1D_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <vector>
#include "Graph/graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_operator.hpp"

namespace netket {

/**
  Transverse field Ising model on an arbitrary graph.
*/

class Ising : public AbstractOperator {
  /**
    Hilbert space descriptor for this hamiltonian.
  */
  std::shared_ptr<const AbstractHilbert> hilbert_;

  std::shared_ptr<const AbstractGraph> graph_;

  const int nspins_;
  double h_;
  double J_;

  /**
    List of bonds for the interaction part.
  */
  std::vector<std::vector<int>> bonds_;

 public:
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

  explicit Ising(std::shared_ptr<const AbstractHilbert> hilbert, double h,
                 double J = 1)
      : hilbert_(hilbert),
        graph_(hilbert->GetGraph()),
        nspins_(hilbert->Size()),
        h_(h),
        J_(J) {
    Init();
  }

  void Init() {
    GenerateBonds();
    InfoMessage() << "Transverse-Field Ising model created " << std::endl;
    InfoMessage() << "h = " << h_ << std::endl;
    InfoMessage() << "J = " << J_ << std::endl;
  }

  /**
    Member function generating the bonds on the lattice.
    bonds[i][k] contains the k-th bond for site i.
  */
  void GenerateBonds() {
    auto adj = graph_->AdjacencyList();

    bonds_.resize(nspins_);

    for (int i = 0; i < nspins_; i++) {
      for (auto s : adj[i]) {
        if (s > i) {
          bonds_[i].push_back(s);
        }
      }
    }
  }

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
  void FindConn(VectorConstRefType v, std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    connectors.clear();
    connectors.resize(nspins_ + 1);
    newconfs.clear();
    newconfs.resize(nspins_ + 1);
    mel.resize(nspins_ + 1);

    mel[0] = 0;
    connectors[0].resize(0);
    newconfs[0].resize(0);

    for (int i = 0; i < nspins_; i++) {
      // spin flips
      mel[i + 1] = -h_;
      connectors[i + 1].push_back(i);
      newconfs[i + 1].push_back(-v(i));

      // interaction part
      for (auto bond : bonds_[i]) {
        mel[0] -= J_ * v(i) * v(bond);
      }
    }
  }

  void ForEachConn(VectorConstRefType v, ConnCallback callback) const override {
    assert(v.size() > 0);

    // local matrix element
    std::complex<double> mel_J = 0;

    // position and value for conf updates
    std::array<int, 1> position;
    std::array<double, 1> value;

    for (int i = 0; i < nspins_; i++) {
      // interaction part
      for (auto bond : bonds_[i]) {
        mel_J -= J_ * v(i) * v(bond);
      }

      // spin-flip
      position[0] = i;
      value[0] = -v(i);
      callback(ConnectorRef{-h_, position, value});
    }

    // diagonal term H(v,v)
    callback(ConnectorRef{mel_J, {}, {}});
  }

  std::shared_ptr<const AbstractHilbert> GetHilbert() const override {
    return hilbert_;
  }
};

}  // namespace netket

#endif
