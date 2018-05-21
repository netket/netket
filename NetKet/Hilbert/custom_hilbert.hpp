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

#ifndef NETKET_CUSTOM_HILBERT_HPP
#define NETKET_CUSTOM_HILBERT_HPP

namespace netket {

/**
  User-Define Hilbert space
*/

class CustomHilbert : public AbstractHilbert {
  std::vector<double> local_;

  int nstates_;

  int size_;

 public:
  explicit CustomHilbert(const json &pars) {
    if (FieldExists(pars["Hilbert"], "QuantumNumbers")) {
      if (!pars["Hilbert"]["QuantumNumbers"].is_array()) {
        std::cerr << "QuantumNumbers is not an array" << std::endl;
        std::abort();
      }
      local_ = pars["Hilbert"]["QuantumNumbers"].get<std::vector<double>>();
    } else {
      std::cerr << "QuantumNumbers are not defined" << std::endl;
    }

    if (FieldExists(pars["Hilbert"], "Size")) {
      size_ = pars["Hilbert"]["Size"];
      if (size_ <= 0) {
        std::cerr << "Hilbert Size parameter must be positive" << std::endl;
        std::abort();
      }
    } else {
      std::cerr << "Hilbert space extent is not defined" << std::endl;
    }

    nstates_ = local_.size();
  }

  bool IsDiscrete() const override { return true; }

  int LocalSize() const override { return nstates_; }

  int Size() const override { return size_; }

  std::vector<double> LocalStates() const override { return local_; }

  void RandomVals(Eigen::VectorXd &state,
                  netket::default_random_engine &rgen) const override {
    std::uniform_int_distribution<int> distribution(0, nstates_ - 1);

    assert(state.size() == size_);

    // unconstrained random
    for (int i = 0; i < state.size(); i++) {
      state(i) = local_[distribution(rgen)];
    }
  }

  void UpdateConf(Eigen::VectorXd &v, const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const override {
    assert(v.size() == size_);

    int i = 0;
    for (auto sf : tochange) {
      v(sf) = newconf[i];
      i++;
    }
  }
};

}  // namespace netket
#endif
