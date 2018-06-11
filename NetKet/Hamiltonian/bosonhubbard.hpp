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
#include <vector>

#include "Utils/exceptions.hpp"
#include "Utils/json_helper.hpp"

#include "abstract_hamiltonian.hpp"

namespace netket {

// Heisenberg model on an arbitrary graph
template <class G>
class BoseHubbard : public AbstractHamiltonian {
  int nsites_;
  double U_;
  double V_;

  double mu_;

  const G &graph_;

  // cutoff in occupation number
  int nmax_;

  // list of bonds for the interaction part
  std::vector<std::vector<int>> bonds_;

  int mynode_;

  /**
    Hilbert space descriptor for this hamiltonian.
  */
  Hilbert hilbert_;

 public:
  // Json constructor
  explicit BoseHubbard(const G &graph, const json &pars)
      : nsites_(graph.Nsites()), graph_(graph)
  {
    nmax_ = FieldVal(pars["Hamiltonian"], "Nmax", "Hamiltonian");
    U_ = FieldVal(pars["Hamiltonian"], "U", "Hamiltonian");

    V_ = FieldOrDefaultVal(pars["Hamiltonian"], "V", .0);
    mu_ = FieldOrDefaultVal(pars["Hamiltonian"], "Mu", .0);

    Init();

    if (FieldExists(pars["Hamiltonian"], "Nbosons")) {
      int nbosons = pars["Hamiltonian"]["Nbosons"];
      SetNbosons(nbosons);
    }
  }

  void Init() {
    GenerateBonds();

    // Specifying the hilbert space
    json hil;
    hil["Hilbert"]["Name"] = "Boson";
    hil["Hilbert"]["Nsites"] = nsites_;
    hil["Hilbert"]["Nmax"] = nmax_;

    hilbert_.Init(hil);

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (mynode_ == 0) {
      std::cout << "# Bose Hubbard model created " << std::endl;
    }
  }

  void SetNbosons(int nbosons) {
    json hil;
    hil["Hilbert"]["Name"] = "Boson";
    hil["Hilbert"]["Nsites"] = nsites_;
    hil["Hilbert"]["Nmax"] = nmax_;
    hil["Hilbert"]["Nbosons"] = nbosons;

    hilbert_.Init(hil);
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

  void FindConn(const Eigen::VectorXd &v,
                std::vector<std::complex<double>> &mel,
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

  const Hilbert &GetHilbert() const override { return hilbert_; }
};

}  // namespace netket

#endif
