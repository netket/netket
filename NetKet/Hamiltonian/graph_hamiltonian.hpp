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
template <class G> class GraphHamiltonian : public AbstractHamiltonian {
  std::vector<LocalOperator> operators_;
  Hilbert hilbert_;

  // Arbitrary graph
  const G &graph_;

  // const std::size_t nvertices_;
  const int nvertices_;

  // Current node node for parallel jobs
  int mynode_;

  /**
  For now the two labels supported are 0 (interacting) and 1 (nearest
  neighbors).

  TODO In the immediate future I plan to add 2 (next-nearest neighbors).
  */

public:
  using MatType = LocalOperator::MatType;

  explicit GraphHamiltonian(const G &graph, const json &pars)
      : hilbert_(pars), graph_(graph), nvertices_(graph.Nsites()) {

    auto pars_hamiltonian = pars["Hamiltonian"];

    // Checking that json contains BondOps, BondColors, and SiteOps
    CheckFieldExists(pars_hamiltonian, "SiteOps");
    if (!pars_hamiltonian["SiteOps"].is_array()) {
      throw InvalidInputError(
          "Hamiltonian: Site operators object is not an array!");
    }

    CheckFieldExists(pars_hamiltonian, "BondOps");
    if (!pars_hamiltonian["BondOps"].is_array()) {
      throw InvalidInputError(
          "Hamiltonian: Bond operators object is not an array!");
    }

    CheckFieldExists(pars_hamiltonian, "BondColors");
    if (!pars_hamiltonian["BondColors"].is_array()) {
      throw InvalidInputError("Hamiltonian.BondColors is not an array");
    }

    // Save operators and bond colors
    auto sop = pars_hamiltonian["SiteOps"].get<std::vector<MatType>>();
    auto bop = pars_hamiltonian["BondOps"].get<std::vector<MatType>>();
    auto op_color = pars_hamiltonian["BondColors"].get<std::vector<int>>();

    // Site operators
    for (int i = 0; i < nvertices_; i++) {
      for (std::size_t j = 0; j < sop.size(); j++) {
        operators_.push_back(
            LocalOperator(hilbert_, sop[j], std::vector<int>{i}));
      }
    }

    // Bond operators

    if (bop.size() != op_color.size()) {
      throw InvalidInputError(
          "The bond Hamiltonian definition is inconsistent."
          "The sizes of BondOps and BondColors do not match.");
    }

    // Get adjacency list from graph
    // auto adj = graph_.AdjacencyList();
    auto ec = graph_.EdgeColors();

    std::cout << "######## Number of edge colors: " << ec.size() << std::endl;

    // Use adj to populate operators
    for (std::map<std::vector<int>, int>::iterator it = ec.begin();
         it != ec.end(); ++it) {
      for (int c = 0; c < op_color.size(); c++) {
        std::cout << c << " " << it->second << " " << (c == it->second)
                  << std::endl;
        if (c == it->second) {
          operators_.push_back(LocalOperator(hilbert_, bop[c], it->first));
          std::cout << it->first[0] << " " << it->first[1] << " " << it->second
                    << std::endl;
        }
      }
    }

    // Interaction
    // std::vector<std::vector<int>> adj_0(nvertices_, std::vector<int>(1));
    // for (int i = 0; i < nvertices_; i++) {
    //   adj_0[i][0] = i;
    // }

    // // Nearest-neighbors
    // auto adj_1 = graph_.AdjacencyList();

    // // std::cout << adj_1.size() << std::endl; // TODO
    //
    // // Next-nearest-neighbors
    // Eigen::MatrixXi adj_2 = Eigen::MatrixXi::Zero(nvertices_, nvertices_);
    // for (std::size_t e1 = 0; e1 < adj_1.size(); e1++) {
    //   for (std::size_t e2 = 0; e2 < adj_1[e1].size(); e2++) {
    //     adj_2(e1, adj_1[e1][e2]) = 1;
    //   }
    // }
    // adj_2 = adj_2 * adj_2;
    // MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);
    // if (mynode_ == 0) {
    //   std::cout << adj_2 << std::endl;
    // }

    // // Use labels and adjacency lists to populate operators_ vector
    // for (std::size_t l = 0; l < op_color.size(); l++) {
    //
    //   // Interaction
    //   if (op_color[l] == 0) {
    //     for (std::size_t s = 0; s < adj_0.size(); s++) {
    //       operators_.push_back(LocalOperator(hilbert_, bop[l], adj_0[s]));
    //     }
    //   }
    //
    //   // Nearest-neighbors
    //   else if (op_color[l] == 1) {
    //     for (int s = 0; s < nvertices_; s++) {
    //       for (auto c : adj_1[s]) {
    //         if (s < c) {
    //           MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);
    //           if (mynode_ == 0) {
    //             std::cout << "NN " << s << " " << c << std::endl;
    //           }
    //           std::vector<int> edge = {s, c};
    //           operators_.push_back(LocalOperator(hilbert_, bop[l], edge));
    //         }
    //       }
    //     }
    //   }
    //
    //   // Next-nearest-neighbors
    //   else if (op_color[l] == 2) {
    //     for (int e1 = 0; e1 < adj_2.rows(); e1++) {
    //       for (int e2 = e1; e2 < adj_2.cols(); e2++) {
    //         if (adj_2(e1, e2) == 1 && e2 > e1) {
    //           std::vector<int> edge = {e1, e2};
    //           operators_.push_back(LocalOperator(hilbert_, bop[l], edge));
    //         }
    //       }
    //     }
    //   }
    //
    //   else {
    //     throw InvalidInputError(
    //         "The label you chose is not currently supported.");
    //   }
    // }

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);
    if (mynode_ == 0) {
      std::cout << "Size of operators_ " << operators_.size() << std::endl;
    }
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
