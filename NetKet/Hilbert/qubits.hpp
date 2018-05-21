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

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "Utils/random_utils.hpp"
#include <vector>
#include "Utils/json_utils.hpp"
#include "abstract_hilbert.hpp"

#ifndef NETKET_QUBITS_HPP
#define NETKET_QUBITS_HPP

namespace netket {

/**
  Hilbert space for qubits.
*/

class Qubit : public AbstractHilbert {
  std::vector<double> local_;

  int nqubits_;

 public:
  explicit Qubit(const json &pars) {
    int nqubits;

    if (!FieldExists(pars["Hilbert"], "Nqubits")) {
      std::cerr << "Nqubits is not defined" << std::endl;
    }

    nqubits = pars["Hilbert"]["Nqubits"];

    Init(nqubits);
  }

  void Init(int nqubits) {
    nqubits_ = nqubits;

    local_.resize(2);

    local_[0] = 0;
    local_[1] = 1;
  }

  bool IsDiscrete() const override { return true; }

  int LocalSize() const override { return 2; }

  int Size() const override { return nqubits_; }

  std::vector<double> LocalStates() const override { return local_; }

  void RandomVals(Eigen::VectorXd &state,
                  netket::default_random_engine &rgen) const override {
    std::uniform_int_distribution<int> distribution(0, 1);

    assert(state.size() == nqubits_);

    // unconstrained random
    for (int i = 0; i < state.size(); i++) {
      state(i) = distribution(rgen);
    }
  }

  void UpdateConf(Eigen::VectorXd &v, const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const override {
    assert(v.size() == nqubits_);

    int i = 0;
    for (auto sf : tochange) {
      v(sf) = newconf[i];
      i++;
    }
  }
};

}  // namespace netket
#endif
