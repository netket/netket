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
#include <Eigen/Dense>
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
  // const int nvertices_;

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

    // Interaction
    std::vector<std::vector<int>> adj_0(nvertices_, std::vector<int>(1));
    for (std::size_t i = 0; i < nvertices_; i++) {
      adj_0[i][0] = i;
      // adj_0[i][1] = i;
    }

    // Nearest-neighbors
    auto adj_1 = graph_.AdjacencyList();
    // std::cout << adj_1.size() << std::endl; // TODO

    // Next-nearest-neighbors
    // TODO make this more efficient
    // std::cout << "nvertices_ " << nvertices_ << std::endl; // TODO
    Eigen::MatrixXi adj_2 = Eigen::MatrixXi::Zero(nvertices_, nvertices_);
    for (std::size_t e1 = 0; e1 < adj_1.size(); e1++) {
      for (std::size_t e2 = 0; e2 < adj_1[e1].size(); e2++) {
        // std::cout << e1 << " " << adj_1[e1][e2] << std::endl; // TODO
        adj_2(e1, adj_1[e1][e2]) = 1;
      }
    }
    adj_2 = adj_2 * adj_2;
    // TODO
    // std::cout << "Printing adj_2\n" << std::endl;
    // std::cout << adj_2 << std::endl;

    // Use labels and adjacency lists to populate operators_ vector
    for (std::size_t l = 0; l < b_label.size(); l++) {
      // Interaction
      if (b_label[l] == 0) {
        for (std::size_t s = 0; s < adj_0.size(); s++) {
          operators_.push_back(LocalOperator(hilbert_, bop[l], adj_0[s]));
        }
      }

      // Nearest-neighbors
      else if (b_label[l] == 1) {
        for (int s = 0; s < adj_1.size(); s++) {
          for (std::size_t c = 0; c < adj_1[s].size(); c++) {
            std::cout << s << " " << adj_1[s][c] << std::endl;
            std::vector<int> edge = {s, adj_1[s][c]};
            operators_.push_back(LocalOperator(hilbert_, bop[l], edge));
          }
        }
      }

      else if (b_label[l] == 2) {
        for (int e1 = 0; e1 < adj_2.rows(); e1++) {
          for (int e2 = 0; e2 < adj_2.cols(); e2++) {
            if (adj_2(e1, e2) == 2) {
              std::vector<int> edge = {e1, e2};
              operators_.push_back(LocalOperator(hilbert_, bop[l], edge));
            }
          }
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
