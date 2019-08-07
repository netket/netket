// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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
#include "Utils/mpi_interface.hpp"

namespace netket {

namespace detail {
void Flipper::RandomState() {
  std::generate(state_.data(), state_.data() + state_.size(), [this]() {
    return local_states_[std::uniform_int_distribution<int>{
        0, static_cast<int>(local_states_.size()) - 1}(engine_)];
  });
}

void Flipper::RandomSites() {
  std::generate(sites_.data(), sites_.data() + sites_.size(), [this]() {
    return std::uniform_int_distribution<Index>{0, Nvisible() - 1}(engine_);
  });
}

void Flipper::RandomNewValues() {
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
        0, static_cast<int>(local_states_.size()) - 2}(engine_);
    return local_states_[idx + (local_states_[idx] >= state_(j, sites_(j)))];
  };
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    new_values_(j) = g(j);
  }
}

Flipper::Flipper(std::pair<Index, Index> const shape,
                 std::vector<double> local_states,
                 default_random_engine& engine)
    : sites_{},
      new_values_{},
      state_{},
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

  // #RandomValues() relies on the fact that locat_states_ are sorted.
  std::sort(local_states_.begin(), local_states_.end());

  sites_.resize(batch_size);
  new_values_.resize(batch_size);
  state_.resize(batch_size, system_size);
  proposed_.resize(batch_size);
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    proposed_[j].values = {&new_values_(j), 1};
    proposed_[j].sites = {&sites_(j), 1};
  }
  Reset();
}

void Flipper::Reset() { RandomState(); }

void Flipper::Update(nonstd::span<const bool> accept) {
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    if (accept[j]) {
      state_(j, sites_(j)) = new_values_(j);
    }
  }
}

nonstd::span<ConfDiff const> Flipper::Propose() {
  using span = nonstd::span<ConfDiff const>;
  RandomSites();
  RandomNewValues();
  return span{proposed_.data(),
              static_cast<span::index_type>(proposed_.size())};
}

const RowMatrix<double>& Flipper::Visible() const noexcept { return state_; }

RowMatrix<double>& Flipper::Visible() noexcept { return state_; }

nonstd::span<const double> Flipper::LocalStates() const noexcept {
  return local_states_;
}

void Flipper::Propose(Eigen::Ref<RowMatrix<double>> x) {
  assert(x.rows() == BatchSize() && x.cols() == Nvisible());
  x = state_;
  const auto updates = Propose();
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    const auto& suggestion = updates[j];
    assert(suggestion.sites.size() == 1 && suggestion.values.size() == 1);
    x(j, suggestion.sites[0]) = suggestion.values[0];
  }
}

}  // namespace detail

MetropolisLocalV2::MetropolisLocalV2(AbstractMachine& machine,
                                     const Index batch_size,
                                     const Index sweep_size,
                                     std::true_type /*safe*/)
    : AbstractSampler{machine},
      flipper_{{batch_size, machine.Nvisible()},
               machine.GetHilbert().LocalStates(),
               GetRandomEngine()},
      proposed_X_(batch_size, machine.Nvisible()),
      proposed_Y_(batch_size),
      current_Y_(batch_size),
      quotient_Y_(batch_size),
      probability_(batch_size),
      accept_(batch_size),
      sweep_size_(sweep_size) {
  GetMachine().LogVal(flipper_.Visible(), current_Y_, {});
}

MetropolisLocalV2::MetropolisLocalV2(AbstractMachine& machine,
                                     const Index batch_size,
                                     const Index sweep_size)
    : MetropolisLocalV2{machine,
                        detail::CheckBatchSize(__FUNCTION__, batch_size),
                        detail::CheckSweepSize(__FUNCTION__, sweep_size),
                        {}} {}

void MetropolisLocalV2::Reset(bool init_random) {
  if (init_random) {
    flipper_.Reset();
    GetMachine().LogVal(flipper_.Visible(), current_Y_, {});
  }
}

std::pair<Eigen::Ref<const RowMatrix<double>>,
          Eigen::Ref<const Eigen::VectorXcd>>
MetropolisLocalV2::CurrentState() const {
  return {flipper_.Visible(), current_Y_};
}

void MetropolisLocalV2::SetVisible(Eigen::Ref<const RowMatrix<double>> x) {
  auto& visible = flipper_.Visible();
  CheckShape(__FUNCTION__, "v", {x.rows(), x.cols()},
             {visible.rows(), visible.cols()});
  const auto local_states = flipper_.LocalStates();
  const auto is_valid = [local_states](const double value) {
    return std::find(local_states.begin(), local_states.end(), value) !=
           local_states.end();
  };
  NETKET_CHECK(std::all_of(x.data(), x.data() + x.size(), std::cref(is_valid)),
               InvalidInputError, "Invalid visible state");
  visible = x;
}

void MetropolisLocalV2::SweepSize(Index const sweep_size) {
  detail::CheckSweepSize(__FUNCTION__, sweep_size);
  sweep_size_ = sweep_size;
}

void MetropolisLocalV2::Next() {
  flipper_.Propose(proposed_X_);  // Now proposed_X_ contains next states `v'`
  GetMachine().LogVal(proposed_X_, /*out=*/proposed_Y_, /*cache=*/{});
  // Calculates acceptance probability
  quotient_Y_ = (proposed_Y_ - current_Y_).exp();
  GetMachineFunc()(quotient_Y_, probability_);
  for (auto i = Index{0}; i < accept_.size(); ++i) {
    accept_(i) = probability_(i) >= 1.0
                     ? true
                     : std::uniform_real_distribution<double>{}(
                           GetRandomEngine()) < probability_(i);
  }
  // Updates current state
  current_Y_ = accept_.select(proposed_Y_, current_Y_);
  flipper_.Update({accept_});

  // Update acceptance counters
  accepted_samples_ += accept_.sum();
  total_samples_ += accept_.size();
}

void MetropolisLocalV2::Sweep() {
  assert(sweep_size_ > 0);
  for (auto i = Index{0}; i < sweep_size_; ++i) {
    Next();
  }
}

}  // namespace netket
