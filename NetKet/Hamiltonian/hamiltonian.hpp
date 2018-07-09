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

#ifndef NETKET_HAMILTONIAN_HPP
#define NETKET_HAMILTONIAN_HPP

#include <memory>

#include "abstract_hamiltonian.hpp"
#include "bosonhubbard.hpp"
#include "custom_hamiltonian.hpp"
#include "heisenberg.hpp"
#include "ising.hpp"

namespace netket {

class Hamiltonian : public AbstractHamiltonian {
  std::unique_ptr<AbstractHamiltonian> h_;

 public:
  explicit Hamiltonian(const Graph &graph, const json &pars) {
    if (!FieldExists(pars, "Hamiltonian")) {
      throw InvalidInputError("Hamiltonian is not defined in the input");
    }

    if (FieldExists(pars["Hamiltonian"], "Name")) {
      if (pars["Hamiltonian"]["Name"] == "Ising") {
        h_.reset(new Ising<Graph>(graph, pars));
      } else if (pars["Hamiltonian"]["Name"] == "Heisenberg") {
        h_.reset(new Heisenberg<Graph>(graph, pars));
      } else if (pars["Hamiltonian"]["Name"] == "BoseHubbard") {
        h_.reset(new BoseHubbard<Graph>(graph, pars));
      } else {
        throw InvalidInputError("Hamiltonian name not found");
      }
    } else {
      h_.reset(new CustomHamiltonian(pars));
    }
  }

  void FindConn(const Eigen::VectorXd &v,
                std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    return h_->FindConn(v, mel, connectors, newconfs);
  }

  const Hilbert &GetHilbert() const override { return h_->GetHilbert(); }
};
}  // namespace netket
#endif
