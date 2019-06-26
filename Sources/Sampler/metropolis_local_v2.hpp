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

#ifndef SOURCES_SAMPLER_METROPOLIS_LOCAL_V2_HPP
#define SOURCES_SAMPLER_METROPOLIS_LOCAL_V2_HPP

#include <limits>
#include <vector>

#include <Eigen/Core>
#include <nonstd/span.hpp>

#include "Utils/random_utils.hpp"

namespace netket {

struct Suggestion {
  nonstd::span<int const> sites;
  nonstd::span<double const> values;
};

class Flipper {
  template <class T>
  using M = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  template <class T>
  using V = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  /// \brief A matrix of size `BatchSize() x SystemSize()`.
  ///
  /// Every row of #sites_ is a permutation of `[0, 1, ..., SystemSize() - 1]`.
  /// We iterate over columns of #sites_ to choose sites at which to change the
  /// quantum numbers.
  M<int> sites_;
  /// \brief A vector of size `BatchSize()` which contains new (proposed)
  /// quantum numbers.
  ///
  /// \pre `values_(j) == local_states_[indices_(j)]` for every
  /// `0 <= j && j < BatchSize()`.
  V<double> values_;
  /// \brief A vector of size `BatchSize()` which contains new (proposed)
  /// indices of quantum numbers.
  ///
  /// It's easier to work with natural number representation of quantum numbers
  /// so we use #indices_ for all internal operations and #values_ for creating
  /// #Suggestion s.
  ///
  /// \pre `values_(j) == local_states_[indices_(j)]` for every
  /// `0 <= j && j < BatchSize()`.
  V<int> indices_;
  /// A matrix of size `BatchSize() x SystemSize()` representing the
  /// current state of #BatchSize different chains.
  M<int> state_;
  /// Current index in the columns of `sites_`.
  int i_;

  std::vector<double> local_states_;

  std::vector<Suggestion> proposed_;

  DistributedRandomEngine engine_;

  int BatchSize() const noexcept { return state_.rows(); }
  int SystemSize() const noexcept { return state_.cols(); }

  nonstd::span<double const> LocalStates() const noexcept {
    using span = nonstd::span<double const>;
    return span{local_states_.data(),
                static_cast<span::index_type>(local_states_.size())};
  }

  void Shuffle() {
    for (auto j = 0; j < BatchSize(); ++j) {
      auto sites = sites_.row(j);
      assert(sites.colStride() == 1);
      std::shuffle(sites.data(), sites.data() + sites.cols(), engine_);
    }
  }

 public:
  void Next(bool accept) {
    if (accept) {
      assert(i_ < SystemSize() && "index out of bounds");
      for (auto j = 0; j < BatchSize(); ++j) {
        assert(0 <= indices_(j) && indices_(j) < LocalStates().size() &&
               "index out of bounds");
        state_(j, sites_(j, i_)) = indices_(j);
      }
    }

    if (++i_ == SystemSize()) {
      Shuffle();
      i_ = 0;
    }
    const auto g = [this](const int j) {
      auto idx = std::uniform_int_distribution<int>{
          0, static_cast<int>(local_states_.size()) - 2}(engine_);
      return idx + (idx >= state_(j, i_));
    };
    for (auto j = 0; j < BatchSize(); ++j) {
      const auto idx = g(j);
      indices_(j) = idx;
      values_(j) = local_states_[static_cast<size_t>(idx)];
      proposed_[j].sites = {&sites_(j, i_), 1};
    }
  }

  nonstd::span<Suggestion const> Read() const noexcept {
    using span = nonstd::span<Suggestion const>;
    return span{proposed_.data(),
                static_cast<span::index_type>(proposed_.size())};
  }
};

}  // namespace netket

#endif  // SOURCES_SAMPLER_METROPOLIS_LOCAL_V2_HPP
