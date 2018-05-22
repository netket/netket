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

#ifndef NETKET_HILBERT_INDEX_HPP
#define NETKET_HILBERT_INDEX_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <vector>
#include "hilbert.hpp"

namespace netket {

class HilbertIndex {
  const std::vector<double> localstates_;

  const int localsize_;

  const int size_;

  std::map<double, int> statenumber_;

  std::vector<std::size_t> basis_;

  int nstates_;

 public:
  explicit HilbertIndex(const Hilbert &hilbert)
      : localstates_(hilbert.LocalStates()),
        localsize_(hilbert.LocalSize()),
        size_(hilbert.Size()) {
    Init();
  }

  void Init() {
    if (size_ * std::log(localsize_) > std::log(MaxStates)) {
      std::cerr << "Hilbert space is too large to be indexed" << std::endl;
      std::abort();
    }

    nstates_ = std::pow(localsize_, size_);

    std::size_t ba = 1;
    for (int s = 0; s < size_; s++) {
      basis_.push_back(ba);
      ba *= localsize_;
    }

    for (std::size_t k = 0; k < localstates_.size(); k++) {
      statenumber_[localstates_[k]] = k;
    }
  }

  // converts a vector of quantum numbers into the unique integer identifier
  std::size_t StateToNumber(const Eigen::VectorXd &v) const {
    std::size_t number = 0;

    for (int i = 0; i < size_; i++) {
      number += statenumber_.at(v(size_ - i - 1)) * basis_[i];
    }

    return number;
  }

  // converts an integer into a vector of quantum numbers
  Eigen::VectorXd NumberToState(std::size_t i) const {
    Eigen::VectorXd result = Eigen::VectorXd::Constant(size_, localstates_[0]);

    std::size_t ip = i;

    int k = size_ - 1;

    while (ip > 0) {
      result(k) = localstates_[ip % localsize_];
      ip /= localsize_;
      k--;
    }
    return result;
  }

  std::size_t NStates() const { return nstates_; }

  constexpr static int MaxStates = std::numeric_limits<int>::max() - 1;
};

}  // namespace netket
#endif
