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
  const AbstractHilbert &hilbert_;
  const AbstractGraph &graph_;

 public:
  using MatType = LocalOperator::MatType;
  using VecType = std::vector<MatType>;

  explicit CustomHamiltonian(const AbstractHilbert &hilbert,
                             const VecType &operators,
                             const std::vector<std::vector<int>> &acting_on)
      : hilbert_(hilbert), graph_(hilbert.GetGraph()) {
    if (acting_on.size() != operators.size()) {
      throw InvalidInputError(
          "The custom Hamiltonian definition is inconsistent: "
          "Check that ActingOn is defined");
    }

    for (std::size_t i = 0; i < operators.size(); i++) {
      operators_.push_back(LocalOperator(hilbert_, operators[i], acting_on[i]));
    }
  }

  // TODO remove
  template <class Ptype>
  explicit CustomHamiltonian(const AbstractHilbert &hilbert, const Ptype &pars)
      : hilbert_(hilbert), graph_(hilbert.GetGraph()) {
    CheckFieldExists(pars, "Operators");
    // TODO
    // if (!pars_hamiltonian["Operators"].is_array()) {
    //   throw InvalidInputError("Hamiltonian: Local operators is not an
    //   array");
    // }

    CheckFieldExists(pars, "ActingOn");
    // if (!pars_hamiltonian["ActingOn"].is_array()) {
    //   throw InvalidInputError("Hamiltonian.ActingOn is not an array");
    // }

    std::vector<MatType> jop =
        FieldVal<std::vector<MatType>>(pars, "Operators");

    std::vector<std::vector<int>> sites =
        FieldVal<std::vector<std::vector<int>>>(pars, "ActingOn");

    if (sites.size() != jop.size()) {
      throw InvalidInputError(
          "The custom Hamiltonian definition is inconsistent: "
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

  const AbstractHilbert &GetHilbert() const override { return hilbert_; }
};
}  // namespace netket
#endif
