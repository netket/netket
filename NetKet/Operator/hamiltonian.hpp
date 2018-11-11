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

#include "Hilbert/hilbert.hpp"
#include "Utils/json_utils.hpp"
#include "Utils/memory_utils.hpp"
#include "abstract_operator.hpp"
#include "bosonhubbard.hpp"
#include "graph_hamiltonian.hpp"
#include "heisenberg.hpp"
#include "ising.hpp"
#include "local_operator.hpp"

namespace netket {

class Hamiltonian : public AbstractOperator {
  std::unique_ptr<AbstractOperator> h_;

 public:
  explicit Hamiltonian(const AbstractHilbert &hilbert, const json &pars) {
    Init(hilbert, pars["Hamiltonian"]);
  }

  template <class Ptype>
  void Init(const AbstractHilbert &hilbert, const Ptype &pars) {
    if (FieldExists(pars, "Name")) {
      std::string name;
      name = FieldVal<std::string>(pars, "Name");

      if (name == "Ising") {
        h_ = netket::make_unique<Ising>(hilbert, pars);
      } else if (name == "Heisenberg") {
        h_ = netket::make_unique<Heisenberg>(hilbert, pars);
      } else if (name == "BoseHubbard") {
        h_ = netket::make_unique<BoseHubbard>(hilbert, pars);
      } else if (name == "Graph") {
        h_ = netket::make_unique<GraphHamiltonian>(hilbert, pars);
      } else {
        std::stringstream s;
        s << "Unknown Hamiltonian type: " << name;
        throw InvalidInputError(s.str());
      }
    } else {
      using MatType = LocalOperator::MatType;
      std::vector<MatType> operators =
          FieldVal<std::vector<MatType>>(pars, "Operators");
      auto acting_on =
          FieldVal<std::vector<std::vector<int>>>(pars, "ActingOn");
      h_ = netket::make_unique<LocalOperator>(hilbert, operators, acting_on);
    }
  }

  void FindConn(const Eigen::VectorXd &v,
                std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    return h_->FindConn(v, mel, connectors, newconfs);
  }

  void ForEachConn(const Eigen::VectorXd &v,
                   ConnCallback callback) const override {
    return h_->ForEachConn(v, callback);
  }

  const AbstractHilbert &GetHilbert() const override {
    return h_->GetHilbert();
  }
};
}  // namespace netket
#endif
