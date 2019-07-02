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

#include "Sampler/metropolis_local_v2.hpp"

namespace netket {

namespace detail {
void Flipper::RandomState() {
  std::generate(state_.data(), state_.data() + state_.size(), [this]() {
    return local_states_[std::uniform_int_distribution<int>{
        0, static_cast<int>(local_states_.size()) - 1}(Generator())];
  });
}

void Flipper::RandomSites() {
  std::generate(sites_.data(), sites_.data() + sites_.size(), [this]() {
    return std::uniform_int_distribution<Index>{0,
                                                SystemSize() - 1}(Generator());
  });
}

void Flipper::RandomValues() {
  // `g` proposes new value for spin `sites_(j)` in Markov chain `j`. There
  // are `local_states_.size() - 1` possible values (minus one is because we
  // don't want to stay in the same state). Thus first, we generate a random
  // number in `[0, local_states_.size() - 2]`. Next step is to transform the
  // result to avoid the gap. Here's an example:
  //
  // ```
  //    indices         0 1 2 3
  //                   +-+-+-+-+
  //    local_states_  | | |X| |
  //                   +-+-+-+-+
  //    transformed
  //      indices       0 1   2
  //
  // ```
  // `X` denotes the current state. We see that transformed index is equal to
  // the original one for all positions before `X`. After `X` however, we need
  // to increment indices by 1.
  const auto g = [this](const int j) {
    const auto idx = std::uniform_int_distribution<int>{
        0, static_cast<int>(local_states_.size()) - 2}(engine_.Get());
    return local_states_[idx + (local_states_[idx] >= state_(j, sites_(j)))];
  };
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    values_(j) = g(j);
  }
}

Flipper::Flipper(std::pair<Index, Index> const shape,
                 std::vector<double> local_states)
    : sites_{},
      values_{},
      state_{},
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

  // #RandomValues() relies on the fact that locat_states_ are sorted.
  std::sort(local_states_.begin(), local_states_.end());

  sites_.resize(batch_size);
  values_.resize(batch_size);
  state_.resize(batch_size, system_size);
  proposed_.resize(batch_size);
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    proposed_[j].values = {&values_(j), 1};
    proposed_[j].sites = {&sites_(j), 1};
  }
  Reset();
}

void Flipper::Reset() {
  RandomState();
  RandomSites();
  RandomValues();
}

void Flipper::Next(nonstd::span<const bool> accept) {
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    if (accept[j]) {
      state_(j, sites_(j)) = values_(j);
    }
  }
  RandomSites();
  RandomValues();
}

nonstd::span<Suggestion const> Flipper::Read() const noexcept {
  using span = nonstd::span<Suggestion const>;
  return span{proposed_.data(),
              static_cast<span::index_type>(proposed_.size())};
}

Eigen::Ref<const Flipper::RowMatrix<double>> Flipper::Current() const noexcept {
  return state_;
}

void Flipper::Read(Eigen::Ref<RowMatrix<double>> x) const noexcept {
  assert(x.rows() == BatchSize() && x.cols() == SystemSize());
  x = state_;
  const auto updates = Read();
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    const auto& suggestion = updates[j];
    assert(suggestion.sites.size() == 1 && suggestion.values.size() == 1);
    x(j, suggestion.sites[0]) = suggestion.values[0];
  }
}

}  // namespace detail

MetropolisLocalV2::MetropolisLocalV2(RbmSpinV2& machine,
                                     AbstractHilbert const& hilbert)
    : forward_{[&machine](Eigen::Ref<const InputType> x) {
        return machine.LogVal(x);
      }},
      flipper_{{machine.BatchSize(), machine.Nvisible()},
               hilbert.LocalStates()},
      proposed_X_(machine.BatchSize(), machine.Nvisible()),
      proposed_Y_(machine.BatchSize()),
      current_Y_(machine.BatchSize()),
      randoms_(machine.BatchSize()),
      accept_(machine.BatchSize()) {
  current_Y_ = forward_(flipper_.Current());
}

void MetropolisLocalV2::Reset() {
  flipper_.Reset();
  current_Y_ = forward_(flipper_.Current());
}

std::pair<Eigen::Ref<const MetropolisLocalV2::InputType>,
          Eigen::Ref<const Eigen::VectorXcd>>
MetropolisLocalV2::Read() {
  return {flipper_.Current(), current_Y_};
}

void MetropolisLocalV2::Next() {
  flipper_.Read(proposed_X_);
  proposed_Y_ = forward_(proposed_X_);
  std::generate(randoms_.data(), randoms_.data() + randoms_.size(), [this]() {
    return std::uniform_real_distribution<double>{}(flipper_.Generator());
  });
  accept_ = randoms_ < (proposed_Y_ - current_Y_).exp().abs().square().min(1.0);
  current_Y_ = accept_.select(proposed_Y_, current_Y_);
  flipper_.Next({accept_});
}

void StepsRange::CheckValid() const {
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
    for (auto j = Index{1}; j < steps.step(); ++j) {
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
