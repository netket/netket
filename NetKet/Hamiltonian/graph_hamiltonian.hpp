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

#include <Eigen/Dense>
#include <array>
#include <unordered_map>
#include <vector>
#include "Utils/json_helper.hpp"
#include "local_operator.hpp"

namespace netket {

// BondHamiltonian on an arbitrary graph
template <class G>
class GraphHamiltonian : public AbstractHamiltonian {
  std::vector<LocalOperator> operators_;
  Hilbert hilbert_;

  // Arbitrary graph
  const G &graph_;

  // const std::size_t nvertices_;
  const int nvertices_;

 public:
  using MatType = LocalOperator::MatType;

  explicit GraphHamiltonian(const G &graph, const json &pars)
      : hilbert_(pars), graph_(graph), nvertices_(graph.Nsites()) {
    auto pars_hamiltonian = pars["Hamiltonian"];

    // Checking that json contains BondOps, BondOpColors, and SiteOps
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

    CheckFieldExists(pars_hamiltonian, "BondOpColors");
    if (!pars_hamiltonian["BondOpColors"].is_array()) {
      throw InvalidInputError("Hamiltonian.BondOpColors is not an array");
    }

    // Save operators and bond colors
    auto sop = pars_hamiltonian["SiteOps"].get<std::vector<MatType>>();
    auto bop = pars_hamiltonian["BondOps"].get<std::vector<MatType>>();
    auto op_color = pars_hamiltonian["BondOpColors"].get<std::vector<int>>();

    // Site operators
    if (sop.size() > 0) {
      for (int i = 0; i < nvertices_; i++) {
        for (std::size_t j = 0; j < sop.size(); j++) {
          operators_.push_back(
              LocalOperator(hilbert_, sop[j], std::vector<int>{i}));
        }
      }
    }

    // Bond operators
    if (bop.size() != op_color.size()) {
      throw InvalidInputError(
          "The bond Hamiltonian definition is inconsistent."
          "The sizes of BondOps and BondOpColors do not match.");
    }

    if (bop.size() > 0) {
      // Use EdgeColors to populate operators
      for (auto const &kv : graph_.EdgeColors()) {
        for (std::size_t c = 0; c < op_color.size(); c++) {
          if (op_color[c] == kv.second && kv.first[0] < kv.first[1]) {
            std::vector<int> edge = {kv.first[0], kv.first[1]};
            operators_.push_back(LocalOperator(hilbert_, bop[c], edge));
          }
        }
      }
    }

    InfoMessage() << "Size of operators_ " << operators_.size() << std::endl;
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
}  // namespace netket
#endif
