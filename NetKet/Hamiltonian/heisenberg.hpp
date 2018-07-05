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

#ifndef NETKET_HEISENBERG_HPP
#define NETKET_HEISENBERG_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "abstract_hamiltonian.hpp"

namespace netket {

// Heisenberg model on an arbitrary graph
template <class G>
class Heisenberg : public AbstractHamiltonian {
  const int nspins_;
  double offdiag_;

  const G &graph_;

  // list of bonds for the interaction part
  std::vector<std::vector<int>> bonds_;

  /**
    Hilbert space descriptor for this hamiltonian.
  */
  Hilbert hilbert_;

 public:
  explicit Heisenberg(const G &graph) : graph_(graph), nspins_(graph.Nsites()) {
    Init();
  }

  // Json constructor
  explicit Heisenberg(const G &graph, const json &pars)
      : nspins_(graph.Nsites()), graph_(graph) {
    Init();

    if (FieldExists(pars["Hamiltonian"], "TotalSz")) {
      double totalsz = pars["Hamiltonian"]["TotalSz"];
      SetTotalSz(totalsz);
    }
  }

  void Init() {
    if (graph_.IsBipartite()) {
      offdiag_ = -2;
    } else {
      offdiag_ = 2;
    }

    GenerateBonds();

    // Specifying the hilbert space
    json hil;
    hil["Hilbert"]["Name"] = "Spin";
    hil["Hilbert"]["Nspins"] = nspins_;
    hil["Hilbert"]["S"] = 0.5;

    hilbert_.Init(hil);

    InfoMessage() << "Heisenberg model created " << std::endl;
  }

  void SetTotalSz(double totalSz) {
    json hil;
    hil["Hilbert"]["Name"] = "Spin";
    hil["Hilbert"]["Nspins"] = nspins_;
    hil["Hilbert"]["S"] = 0.5;
    hil["Hilbert"]["TotalSz"] = totalSz;

    hilbert_.Init(hil);
  }

  void GenerateBonds() {
    auto adj = graph_.AdjacencyList();

    bonds_.resize(nspins_);

    for (int i = 0; i < nspins_; i++) {
      for (auto s : adj[i]) {
        if (s > i) {
          bonds_[i].push_back(s);
        }
      }
    }
  }

  void FindConn(const Eigen::VectorXd &v,
                std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    connectors.clear();
    connectors.resize(1);
    newconfs.clear();
    newconfs.resize(1);
    mel.resize(1);

    // computing interaction part Sz*Sz
    mel[0] = 0.;
    connectors[0].resize(0);
    newconfs[0].resize(0);

    for (int i = 0; i < nspins_; i++) {
      for (auto bond : bonds_[i]) {
        // interaction part
        mel[0] += v(i) * v(bond);

        // spin flips
        if (v(i) != v(bond)) {
          connectors.push_back(std::vector<int>({i, bond}));
          newconfs.push_back(std::vector<double>({v(bond), v(i)}));
          mel.push_back(offdiag_);
        }
      }
    }
  }

  const Hilbert &GetHilbert() const override { return hilbert_; }
};

}  // namespace netket

#endif
