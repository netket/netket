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

#include "local_operator.hh"
#include <vector>

namespace netket {

class CustomHamiltonian : public AbstractHamiltonian {

  std::vector<LocalOperator> operators_;
  Hilbert hilbert_;

public:
  using MatType = LocalOperator::MatType;

  explicit CustomHamiltonian(const json &pars) : hilbert_(pars) {

    if (!FieldExists(pars["Hamiltonian"], "Operators")) {
      std::cerr << "Local operators in the Hamiltonian are not defined"
                << std::endl;
      std::abort();
    }

    if (!pars["Hamiltonian"]["Operators"].is_array()) {
      std::cerr << "Local operators is not an array" << std::endl;
      std::abort();
    }

    if (!FieldExists(pars["Hamiltonian"], "ActingOn")) {
      std::cerr << "Local operators support in the Hamiltonian is not defined"
                << std::endl;
      std::abort();
    }

    if (!pars["Hamiltonian"]["ActingOn"].is_array()) {
      std::cerr << "ActingOn is not an array" << std::endl;
      std::abort();
    }

    auto jop = pars["Hamiltonian"]["Operators"].get<std::vector<MatType>>();
    auto sites =
        pars["Hamiltonian"]["ActingOn"].get<std::vector<std::vector<int>>>();

    if (sites.size() != jop.size()) {
      std::cerr << "The custom Hamiltonian definition is inconsistent:"
                << std::endl;
      std::cerr << "Check that ActingOn is defined" << std::endl;
      std::abort();
    }

    for (std::size_t i = 0; i < jop.size(); i++) {
      operators_.push_back(LocalOperator(hilbert_, jop[i], sites[i]));
    }
  }

  void FindConn(const Eigen::VectorXd &v,
                std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) {
    connectors.clear();
    newconfs.clear();
    mel.resize(0);

    for (std::size_t i = 0; i < operators_.size(); i++) {
      operators_[i].AddConn(v, mel, connectors, newconfs);
    }
  }

  const Hilbert &GetHilbert() const { return hilbert_; }
};
} // namespace netket
#endif
