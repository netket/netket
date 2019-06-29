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

#include "Machine/rbm_spin_v2.hpp"
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
  Eigen::Matrix<double, D, D, Eigen::RowMajor> state_;

  /// Current index in the columns of `sites_`.
  Index i_;

  std::vector<double> local_states_;

  std::vector<Suggestion> proposed_;

  DistributedRandomEngine engine_;

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
      std::shuffle(sites.data(), sites.data() + sites.cols(), Generator());
    }
  }

  /// Randomizes the state.
  void RandomState() {
    std::generate(state_.data(), state_.data() + state_.size(), [this]() {
      return std::uniform_int_distribution<int>{
          0, static_cast<int>(local_states_.size()) - 1}(Generator());
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
  Flipper(std::pair<Index, Index> const shape, std::vector<double> local_states)
      : sites_{},
        values_{},
        state_{},
        i_{0},
        local_states_{std::move(local_states)},
        proposed_{},
        engine_{} {
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

  Index BatchSize() const noexcept { return state_.rows(); }
  Index SystemSize() const noexcept { return state_.cols(); }
  default_random_engine& Generator() noexcept { return engine_.Get(); }

  void Reset() {
    RandomState();
    i_ = 0;
    Shuffle();
    const auto g = [this](const int j) {
      const auto idx = std::uniform_int_distribution<int>{
          0, static_cast<int>(local_states_.size()) - 2}(engine_.Get());
      return local_states_[idx + (local_states_[idx] >= state_(j, i_))];
    };
    for (auto j = Index{0}; j < BatchSize(); ++j) {
      values_(j) = g(j);
      proposed_[j].sites = {&sites_(j, i_), 1};
    }
  }

  void Next(nonstd::span<const bool> accept) {
    assert(i_ < SystemSize() && "index out of bounds");
    for (auto j = Index{0}; j < BatchSize(); ++j) {
      if (accept[j]) {
        state_(j, sites_(j, i_)) = values_(j);
      }
    }

    if (++i_ == SystemSize()) {
      Shuffle();
      i_ = 0;
    }
    const auto g = [this](const int j) {
      const auto idx = std::uniform_int_distribution<int>{
          0, static_cast<int>(local_states_.size()) - 2}(engine_.Get());
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

  Eigen::Ref<const Eigen::MatrixXd> Current() const { return state_; }

  void Read(Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
                x) const noexcept {
    assert(x.rows() == BatchSize() && x.cols() == SystemSize());
    x = state_;
    const auto updates = Read();
    for (auto j = Index{0}; j < BatchSize(); ++j) {
      const auto& suggestion = updates[j];
      assert(suggestion.sites.size() == 1 && suggestion.values.size() == 1);
      x(j, suggestion.sites[0]) = suggestion.values[0];
    }
  }
};
}  // namespace detail

class MetropolisLocalV2 {
  using InputType = Eigen::Ref<const Eigen::MatrixXd>;
  using ForwardFn = std::function<auto(Eigen::Ref<const Eigen::MatrixXd>)
                                      ->Eigen::Ref<const Eigen::VectorXcd>>;

  ForwardFn forward_;
  detail::Flipper flipper_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      proposed_X_;
  Eigen::ArrayXcd proposed_Y_;
  Eigen::ArrayXcd current_Y_;
  Eigen::ArrayXd randoms_;
  Eigen::Array<bool, Eigen::Dynamic, 1> accept_;

 public:
  MetropolisLocalV2(RbmSpinV2& machine, AbstractHilbert const& hilbert)
      : forward_{[&machine](InputType x) { return machine.LogVal(x); }},
        flipper_{{machine.BatchSize(), machine.Nvisible()},
                 hilbert.LocalStates()},
        proposed_X_(machine.BatchSize(), machine.Nvisible()),
        proposed_Y_(machine.BatchSize()),
        current_Y_(machine.BatchSize()),
        randoms_(machine.BatchSize()),
        accept_(machine.BatchSize()) {
    current_Y_ = forward_(flipper_.Current());
  }

  void Reset() {
    flipper_.Reset();
    current_Y_ = forward_(flipper_.Current());
  }

  std::pair<Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor>>,
            Eigen::Ref<const Eigen::VectorXcd>>
  Read() {
    return {flipper_.Current(), current_Y_};
  }

  Index BatchSize() const noexcept { return flipper_.BatchSize(); }
  Index SystemSize() const noexcept { return flipper_.SystemSize(); }

  void Next() {
    flipper_.Read(proposed_X_);
    proposed_Y_ = forward_(proposed_X_);
    std::generate(randoms_.data(), randoms_.data() + randoms_.size(), [this]() {
      return std::uniform_real_distribution<double>{}(flipper_.Generator());
    });
    accept_ =
        randoms_ < (proposed_Y_ - current_Y_).real().exp().square().min(1.0);
    current_Y_ = accept_.select(proposed_Y_, current_Y_);
    flipper_.Next({accept_});
  }
};

struct StepsRange {
  StepsRange(std::tuple<Index, Index, Index> const& steps)
      : start_{std::get<0>(steps)},
        end_{std::get<1>(steps)},
        step_{std::get<2>(steps)} {
    CheckValid();
  }

  constexpr Index start() const noexcept { return start_; }
  constexpr Index end() const noexcept { return end_; }
  constexpr Index step() const noexcept { return step_; }

  constexpr Index size() const noexcept {
    return (end_ - start_ - 1) / step_ + 1;
  }

 private:
  void CheckValid() const {
    const auto error = [this](const char* expected) {
      std::ostringstream msg;
      msg << "invalid steps range: (start=" << start_ << ", end=" << end_
          << ", step=" << step_ << "); " << expected;
      throw InvalidInputError{msg.str()};
    };
    if (start_ < 0) error("expected start >= 0");
    if (end_ < 0) error("expected end >= 0");
    if (step_ <= 0) error("expected step >= 1");
    if (end_ < start_) error("expected start <= end");
  }

  Index start_;
  Index end_;
  Index step_;
};  // namespace netket

namespace detail {
template <class Skip, class Record>
void LoopV2(StepsRange const& steps, Skip&& skip, Record&& record) {
  auto i = Index{0};
  // Skipping [0, 1, ..., start)
  for (; i < steps.start(); ++i) {
    skip();
  }
  // Record [start]
  record();
  for (i += steps.step(); i < steps.end(); i += steps.step()) {
    for (auto j = Index{0}; j < steps.step(); ++j) {
      skip();
    }
    record();
  }
}
}  // namespace detail

std::tuple<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::VectorXcd>
ComputeSamples(MetropolisLocalV2& sampler, StepsRange const& steps) {
  sampler.Reset();
  const auto num_samples = steps.size() * sampler.BatchSize();

  using Matrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  Matrix samples(num_samples, sampler.SystemSize());
  Eigen::VectorXcd values(num_samples);

  struct Skip {
    MetropolisLocalV2& sampler_;
    void operator()() const { sampler_.Next(); }
  };

  struct Record {
    MetropolisLocalV2& sampler_;
    Matrix& samples_;
    VectorXcd& values_;
    Index i_;

    std::pair<Eigen::Ref<Matrix>, Eigen::Ref<VectorXcd>> Batch() {
      const auto n = sampler_.BatchSize();
      return {samples_.block(i_ * n, 0, n, samples_.cols()),
              values_.segment(i_ * n, n)};
    }

    void operator()() {
      assert(i_ * sampler_.BatchSize() < samples_.rows());
      Batch() = sampler_.Read();
      if (++i_ * sampler_.BatchSize() != samples_.rows()) {
        sampler_.Next();
      }
    }
  };

  detail::LoopV2(steps, Skip{sampler}, Record{sampler, samples, values, 0});
  return std::make_tuple(std::move(samples), std::move(values));
}

}  // namespace netket

#endif  // SOURCES_SAMPLER_METROPOLIS_LOCAL_V2_HPP
