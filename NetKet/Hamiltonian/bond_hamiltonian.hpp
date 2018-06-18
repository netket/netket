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

#ifndef NETKET_BOND_HAMILTONIAN_CC
#define NETKET_BOND_HAMILTONIAN_CC

#include "Utils/json_helper.hpp"
#include "local_operator.hpp"
#include <iostream> // TODO remove
#include <vector>

namespace netket {

// BondHamiltonian on an arbitrary graph
template <class G> class BondHamiltonian : public AbstractHamiltonian {
  std::vector<LocalOperator> operators_;
  Hilbert hilbert_;

  // Arbitrary graph
  const G &graph_;

  const std::size_t nvertices_;

  int mynode_;

  /**
  For now the two labels supported are 0 (interacting) and 1 (nearest
  neighbors).

  TODO In the immediate future I plan to add 2 (next-nearest neighbors).
  */

  // // Bond operators
  // std::vector<LocalOperator::MatType> bop_;
  //
  // // Labels distinguishing which bonds the operators act on
  // std::vector<int> b_label_;
  //
  // // Couplings corresponding to b_label_ and bop_
  // std::vector<double> b_couple_; // TODO Should this be complex?

public:
  using MatType = LocalOperator::MatType;

  explicit BondHamiltonian(const G &graph, const json &pars)
      : hilbert_(pars), graph_(graph), nvertices_(graph.Nsites()) {

    auto pars_hamiltonian = pars["Hamiltonian"];

    // Checking that json contains BondOps, BondLabels, and BondCoupling
    CheckFieldExists(pars_hamiltonian, "BondOps");
    if (!pars_hamiltonian["BondOps"].is_array()) {
      throw InvalidInputError(
          "Hamiltonian: Bond operators object is not an array!");
    }

    CheckFieldExists(pars_hamiltonian, "BondLabels");
    if (!pars_hamiltonian["BondLabels"].is_array()) {
      throw InvalidInputError("Hamiltonian.BondLabels is not an array");
    }

    CheckFieldExists(pars_hamiltonian, "BondCoupling");
    if (!pars_hamiltonian["BondCoupling"].is_array()) {
      throw InvalidInputError("Hamiltonian.BondCoupling is not an array");
    }

    auto bop = pars_hamiltonian["BondOps"].get<std::vector<MatType>>();
    auto b_label = pars_hamiltonian["BondLabels"].get<std::vector<int>>();
    auto b_couple = pars_hamiltonian["BondCoupling"].get<std::vector<double>>();

    // auto jop = pars_hamiltonian["Operators"].get<std::vector<MatType>>();
    // auto sites =
    //     pars_hamiltonian["ActingOn"].get<std::vector<std::vector<int>>>();

    // if (sites.size() != jop.size()) {
    //   throw InvalidInputError(
    //       "The custom Hamiltonian definition is inconsistent: "
    //       "Check that ActingOn is defined");
    // }

    if (bop.size() != b_label.size() || b_label.size() != b_couple.size() ||
        bop.size() != b_couple.size()) {
      throw InvalidInputError(
          "The bond Hamiltonian definition is inconsistent."
          "The sizes of BondOps, BondLabels, and BondCoupling do not match.");
    }

    // Create interacting site list
    std::vector<std::vector<int>> adj_0(nvertices_, std::vector<int>(1));
    for (std::size_t i = 0; i < nvertices_; i++) {
      adj_0[i][0] = i;
      // adj_0[i][1] = i;
    }

    // json hil;
    // hil["Hilbert"]["Name"] = "Spin";
    // hil["Hilbert"]["Nspins"] = nvertices_;
    // hil["Hilbert"]["S"] = 0.5;
    // hilbert_.Init(hil);

    auto adj_1 = graph_.AdjacencyList();
    std::cout << adj_1.size() << std::endl;

    for (std::size_t l = 0; l < b_label.size(); l++) {
      // Interaction
      if (b_label[l] == 0) {
        for (std::size_t s = 0; s < adj_0.size(); s++) {
          operators_.push_back(LocalOperator(hilbert_, bop[l], adj_0[s]));
        }
      }

      else if (b_label[l] == 1) {
        for (std::size_t s = 0; s < adj_1.size(); s++) {
          std::cout << "Label 1, site " << s << std::endl;
          std::cout << adj_1[0].size() << std::endl;
          std::cout << adj_1[0][0] << " " << adj_1[0][1] << std::endl;
          operators_.push_back(LocalOperator(hilbert_, bop[l], adj_1[s]));
        }
      }

      else {
        throw InvalidInputError(
            "The label you chose is not currently supported.");
      }
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);
    if (mynode_ == 0) {
      std::cout << "Size of operators_ " << operators_.size() << std::endl;
    }

    // for (std::size_t i = 0; i < jop.size(); i++) {
    //   operators_.push_back(LocalOperator(hilbert_, jop[i], sites[i]));
    // }
  }

  void FindConn(const Eigen::VectorXd &v,
                std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    connectors.clear();
    newconfs.clear();
    mel.resize(0);

    for (std::size_t i = 0; i < operators_.size(); i++) {
      operators_[i].AddConn(v, mel, connectors, newconfs);
    }
  }

  const Hilbert &GetHilbert() const override { return hilbert_; }
};
} // namespace netket
#endif
