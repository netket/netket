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

#include "Utils/exceptions.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

struct Suggestion {
  nonstd::span<Index const> sites;
  nonstd::span<double const> values;
};

namespace detail {
class Flipper {
  template <class T>
  using M = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  template <class T>
  using V = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  static constexpr auto D = Eigen::Dynamic;

  /// \brief A matrix of size `BatchSize() x SystemSize()`.
  ///
  /// Every row of #sites_ is a permutation of `[0, 1, ..., SystemSize() - 1]`.
  /// We iterate over columns of #sites_ to choose sites at which to change the
  /// quantum numbers.
  Eigen::Matrix<Index, D, D, Eigen::RowMajor> sites_;
  /// \brief A vector of size `BatchSize()` which contains new (proposed)
  /// quantum numbers.
  ///
  /// \pre `values_(j) == local_states_[indices_(j)]` for every
  /// `0 <= j && j < BatchSize()`.
  Eigen::Matrix<double, D, 1> values_;
  /// A matrix of size `BatchSize() x SystemSize()` representing the
  /// current state of #BatchSize different chains.
  Eigen::Matrix<double, D, D, Eigen::ColMajor> state_;

  /// Current index in the columns of `sites_`.
  Index i_;

  std::vector<double> local_states_;

  std::vector<Suggestion> proposed_;

  default_random_engine& engine_;

  Index BatchSize() const noexcept { return state_.rows(); }
  Index SystemSize() const noexcept { return state_.cols(); }

  /// Returns local states of the Hilbert space as a `span`.
  nonstd::span<double const> LocalStates() const noexcept {
    using span = nonstd::span<double const>;
    return span{local_states_.data(),
                static_cast<span::index_type>(local_states_.size())};
  }

  /// \brief Shuffles every row of the #sites_ matrix.
  ///
  /// This is used for choosing which quantum numbers to update.
  void Shuffle() {
    for (auto j = 0; j < BatchSize(); ++j) {
      auto sites = sites_.row(j);
      assert(sites.colStride() == 1);
      std::shuffle(sites.data(), sites.data() + sites.cols(), engine_);
    }
  }

  /// Randomizes the state.
  void RandomState() {
    std::generate(state_.data(), state_.data() + state_.size(), [this]() {
      return std::uniform_int_distribution<int>{
          0, static_cast<int>(local_states_.size()) - 1}(engine_);
    });
  }

  void ResetState() noexcept {
    static_assert(decltype(sites_)::IsRowMajor == true, "");
    std::iota(&sites_(0, 0), &sites_(0, sites_.cols()), Index{0});
    for (auto j = Index{1}; j < sites_.rows(); ++j) {
      sites_.row(j) = sites_.row(0);
    }
  }

 public:
  Flipper(std::pair<Index, Index> const shape, std::vector<double> local_states,
          default_random_engine& engine)
      : sites_{},
        values_{},
        state_{},
        i_{0},
        local_states_{std::move(local_states)},
        proposed_{},
        engine_{engine} {
    Index batch_size, system_size;
    std::tie(batch_size, system_size) = shape;
    if (batch_size < 1) {
      std::ostringstream msg;
      msg << "invalid batch size: " << batch_size << "; expected >=1";
      throw InvalidInputError{msg.str()};
    }
    if (system_size < 1) {
      std::ostringstream msg;
      msg << "invalid system size: " << system_size << "; expected >=1";
      throw InvalidInputError{msg.str()};
    }
    if (local_states_.empty()) {
      throw InvalidInputError{"invalid local states: []"};
    }

    std::sort(local_states_.begin(), local_states_.end());

    sites_.resize(batch_size, system_size);
    values_.resize(batch_size);
    state_.resize(batch_size, system_size);
    proposed_.resize(batch_size);
    for (auto j = Index{0}; j < BatchSize(); ++j) {
      proposed_[j].values = {&values_(j), 1};
    }

    ResetState();
    Reset();
  }

  void Reset() {
    RandomState();
    i_ = SystemSize() - 1;
    Next(false);
  }

  void Next(bool accept) {
    if (accept) {
      assert(i_ < SystemSize() && "index out of bounds");
      for (auto j = Index{0}; j < BatchSize(); ++j) {
        state_(j, sites_(j, i_)) = values_(j);
      }
    }

    if (++i_ == SystemSize()) {
      Shuffle();
      i_ = 0;
    }
    const auto g = [this](const int j) {
      const auto idx = std::uniform_int_distribution<int>{
          0, static_cast<int>(local_states_.size()) - 2}(engine_);
      return local_states_[idx + (local_states_[idx] >= state_(j, i_))];
    };
    for (auto j = Index{0}; j < BatchSize(); ++j) {
      values_(j) = g(j);
      proposed_[j].sites = {&sites_(j, i_), 1};
    }
  }

  nonstd::span<Suggestion const> Read() const noexcept {
    using span = nonstd::span<Suggestion const>;
    return span{proposed_.data(),
                static_cast<span::index_type>(proposed_.size())};
  }
};
}  // namespace detail

}  // namespace netket

#endif  // SOURCES_SAMPLER_METROPOLIS_LOCAL_V2_HPP
