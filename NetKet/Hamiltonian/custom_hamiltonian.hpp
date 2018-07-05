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

#ifndef NETKET_CUSTOM_HAMILTONIAN_CC
#define NETKET_CUSTOM_HAMILTONIAN_CC

#include <vector>

#include "Utils/json_helper.hpp"

#include "local_operator.hpp"

namespace netket {

class CustomHamiltonian : public AbstractHamiltonian {
  std::vector<LocalOperator> operators_;
  Hilbert hilbert_;

 public:
  using MatType = LocalOperator::MatType;

  explicit CustomHamiltonian(const json &pars) : hilbert_(pars)
  {
    auto pars_hamiltonian = pars["Hamiltonian"];

    CheckFieldExists(pars_hamiltonian, "Operators");
    if (!pars_hamiltonian["Operators"].is_array()) {
      throw InvalidInputError("Hamiltonian: Local operators is not an array");
    }

    CheckFieldExists(pars_hamiltonian, "ActingOn");
    if (!pars_hamiltonian["ActingOn"].is_array()) {
      throw InvalidInputError("Hamiltonian.ActingOn is not an array");
    }

    auto jop = pars_hamiltonian["Operators"].get<std::vector<MatType>>();
    auto sites =
        pars_hamiltonian["ActingOn"].get<std::vector<std::vector<int>>>();

    if (sites.size() != jop.size()) {
      throw InvalidInputError("The custom Hamiltonian definition is inconsistent: "
                              "Check that ActingOn is defined");
    }

    for (std::size_t i = 0; i < jop.size(); i++) {
      operators_.push_back(LocalOperator(hilbert_, jop[i], sites[i]));
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
}  // namespace netket
#endif
