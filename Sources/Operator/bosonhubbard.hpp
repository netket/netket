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

#ifndef NETKET_BOSONHUBBARD_HPP
#define NETKET_BOSONHUBBARD_HPP

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "Graph/graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"
#include "Utils/exceptions.hpp"
#include "Utils/messages.hpp"
#include "abstract_operator.hpp"

namespace netket {

// Heisenberg model on an arbitrary graph
class BoseHubbard : public AbstractOperator {
  const AbstractGraph &graph_;

  int nsites_;

  // cutoff in occupation number
  int nmax_;

  double U_;
  double V_;

  double mu_;

  // list of bonds for the interaction part
  std::vector<std::vector<int>> bonds_;

 public:
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

  BoseHubbard(std::shared_ptr<const AbstractHilbert> hilbert, double U,
              double V = 0., double mu = 0.)
      : AbstractOperator(hilbert),
        graph_(hilbert->GetGraph()),
        nsites_(hilbert->Size()),
        U_(U),
        V_(V),
        mu_(mu) {
    nmax_ = hilbert->LocalSize() - 1;
    Init();
  }

  void Init() {
    GenerateBonds();
    InfoMessage() << "Bose Hubbard model created \n";
    InfoMessage() << "U= " << U_ << std::endl;
    InfoMessage() << "V= " << V_ << std::endl;
    InfoMessage() << "mu= " << mu_ << std::endl;
    InfoMessage() << "Nmax= " << nmax_ << std::endl;
  }

  void GenerateBonds() {
    auto adj = graph_.AdjacencyList();

    bonds_.resize(nsites_);

    for (int i = 0; i < nsites_; i++) {
      for (auto s : adj[i]) {
        if (s > i) {
          bonds_[i].push_back(s);
        }
      }
    }
  }

  void FindConn(VectorConstRefType v, std::vector<Complex> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    connectors.clear();
    connectors.resize(1);
    newconfs.clear();
    newconfs.resize(1);
    mel.resize(1);

    mel[0] = 0.;
    connectors[0].resize(0);
    newconfs[0].resize(0);

    for (int i = 0; i < nsites_; i++) {
      // chemical potential
      mel[0] -= mu_ * v(i);

      // on-site interaction
      mel[0] += 0.5 * U_ * v(i) * (v(i) - 1);

      for (auto bond : bonds_[i]) {
        // nn interaction
        mel[0] += V_ * v(i) * v(bond);
        // hopping
        if (v(i) > 0 && v(bond) < nmax_) {
          connectors.push_back(std::vector<int>({i, bond}));
          newconfs.push_back(std::vector<double>({v(i) - 1, v(bond) + 1}));
          mel.push_back(-std::sqrt(v(i)) * std::sqrt(v(bond) + 1));
        }
        if (v(bond) > 0 && v(i) < nmax_) {
          connectors.push_back(std::vector<int>({bond, i}));
          newconfs.push_back(std::vector<double>({v(bond) - 1, v(i) + 1}));
          mel.push_back(-std::sqrt(v(bond)) * std::sqrt(v(i) + 1));
        }
      }
    }
  }
};

}  // namespace netket

#endif
