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
#include <memory>
#include <nonstd/span.hpp>
#include <vector>

#include "Utils/next_variation.hpp"

namespace netket {

class HilbertIndex {
 public:
  HilbertIndex(std::vector<double> localstates, int local_size, int size)
      : localstates_(std::move(localstates)),
        localsize_(local_size),
        size_(size) {
    Init();
  }

  void Init() {
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
      assert(statenumber_.count(v(size_ - i - 1)) > 0);
      number += statenumber_.at(v(size_ - i - 1)) * basis_[i];
    }

    return number;
  }

  // converts a vector of quantum numbers into the unique integer identifier
  // this version assumes that a number representaiton is already known for the
  // given vector v, and this function is used to update it
  std::size_t DeltaStateToNumber(const Eigen::VectorXd &v,
                                 nonstd::span<const int> connector,
                                 nonstd::span<const double> newconf) const {
    std::size_t number = 0;

    for (int k = 0; k < connector.size(); k++) {
      const int ich = connector[k];
      assert(statenumber_.count(v(ich)) > 0);
      assert(statenumber_.count(newconf[k]) > 0);
      number -= statenumber_.at(v(ich)) * basis_[size_ - ich - 1];
      number += statenumber_.at(newconf[k]) * basis_[size_ - ich - 1];
    }

    return number;
  }

  // converts an integer into a vector of quantum numbers
  Eigen::VectorXd NumberToState(int i) const {
    Eigen::VectorXd result = Eigen::VectorXd::Constant(size_, localstates_[0]);

    int ip = i;

    int k = size_ - 1;

    while (ip > 0) {
      assert((ip % localsize_) < localstates_.size());
      result(k) = localstates_[ip % localsize_];
      ip /= localsize_;
      k--;
    }
    return result;
  }

  int NStates() const { return nstates_; }

  constexpr static int MaxStates = std::numeric_limits<int>::max() - 1;

 private:
  const std::vector<double> localstates_;
  const int localsize_;
  const int size_;
  std::map<double, int> statenumber_;
  std::vector<std::size_t> basis_;
  int nstates_;
};

class StateIterator {
 public:
  // typedefs required for iterators
  using iterator_category = std::input_iterator_tag;
  using difference_type = Index;
  using value_type = Eigen::VectorXd;
  using pointer_type = Eigen::VectorXd *;
  using reference_type = Eigen::VectorXd &;

  explicit StateIterator(const HilbertIndex &index) : i_(0), index_(index) {}

  value_type operator*() const { return index_.NumberToState(i_); }

  StateIterator &operator++() {
    ++i_;
    return *this;
  }

  // TODO(C++17): Replace with comparison to special Sentinel type, since
  // C++17 allows end() to return a different type from begin().
  bool operator!=(const StateIterator &) { return i_ < index_.NStates(); }
  // pybind11::make_iterator requires operator==
  bool operator==(const StateIterator &other) { return !(*this != other); }

  StateIterator begin() const { return *this; }
  StateIterator end() const { return *this; }

 private:
  int i_;
  const HilbertIndex &index_;
};

}  // namespace netket
#endif
