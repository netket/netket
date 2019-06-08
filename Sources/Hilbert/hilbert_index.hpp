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

#include <limits>
#include <map>
#include <nonstd/span.hpp>
#include <vector>

#include <Eigen/Dense>

#include "common_types.hpp"

namespace netket {

class HilbertIndex {
 public:
  HilbertIndex(std::vector<double> localstates, int local_size, int size);

  // converts a vector of quantum numbers into the unique integer identifier
  std::size_t StateToNumber(const Eigen::VectorXd &v) const;

  // converts a vector of quantum numbers into the unique integer identifier
  // this version assumes that a number representaiton is already known for the
  // given vector v, and this function is used to update it
  std::size_t DeltaStateToNumber(const Eigen::VectorXd &v,
                                 nonstd::span<const int> connector,
                                 nonstd::span<const double> newconf) const;

  // converts an integer into a vector of quantum numbers
  Eigen::VectorXd NumberToState(int i) const;

  /*constexpr*/ int NStates() const noexcept { return nstates_; }
  constexpr static int MaxStates = std::numeric_limits<int>::max() - 1;

 private:
  void Init();

  const std::vector<double> localstates_;
  const int localsize_;
  const int size_;
  std::map<double, int> statenumber_;
  std::vector<std::size_t> basis_;
  int nstates_;
};

class StateGenerator {
 public:
  using difference_type = Index;
  using value_type = Eigen::VectorXd;
  using reference = value_type;

  explicit StateGenerator(const HilbertIndex &index) : i_(0), index_(index) {}

  reference operator*() const { return index_.NumberToState(i_); }

  StateGenerator &operator++() {
    ++i_;
    return *this;
  }

  // TODO(C++17): Replace with comparison to special Sentinel type, since
  // C++17 allows end() to return a different type from begin().
  bool operator!=(const StateGenerator &) { return i_ < index_.NStates(); }
  // pybind11::make_iterator requires operator==
  bool operator==(const StateGenerator &other) { return !(*this != other); }

  StateGenerator begin() const { return *this; }
  StateGenerator end() const { return *this; }

 private:
  int i_;
  const HilbertIndex &index_;
};

}  // namespace netket

#endif  // NETKET_HILBERT_INDEX_HPP
