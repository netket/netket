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

#include "Sampler/metropolis_hastings.hpp"

namespace netket {

MetropolisHastings::MetropolisHastings(
    AbstractMachine& machine,
    MetropolisHastings::TransitionKernel transition_kernel, Index n_chains,
    Index sweep_size, Index batch_size)
    : AbstractSampler(machine),
      transition_kernel_(transition_kernel),
      n_chains_(n_chains),
      sweep_size_(sweep_size),
      batch_size_(batch_size) {
  detail::CheckNChains(__FUNCTION__, n_chains);
  detail::CheckSweepSize(__FUNCTION__, sweep_size);
  detail::CheckBatchSize(__FUNCTION__, batch_size, n_chains);

  current_X_.resize(n_chains, machine.Nvisible());
  proposed_X_.resize(n_chains, machine.Nvisible());
  current_Y_.resize(n_chains);
  proposed_Y_.resize(n_chains);
  quotient_Y_.resize(n_chains);
  probability_.resize(n_chains);
  accept_.resize(n_chains);
  log_acceptance_correction_.resize(n_chains);

  Reset(true);
}

void MetropolisHastings::Reset(bool init_random) {
  if (init_random) {
    for (Index i = 0; i < n_chains_; i++) {
      GetMachine().GetHilbert().RandomVals(current_X_.row(i),
                                           GetRandomEngine());
    }
  }
  LogValBatched(current_X_, current_Y_, {});

  accepted_samples_ = 0;
  total_samples_ = 0;
}

std::pair<Eigen::Ref<const RowMatrix<double>>,
          Eigen::Ref<const Eigen::VectorXcd>>
MetropolisHastings::CurrentState() const {
  return {current_X_, current_Y_};
}

void MetropolisHastings::SetVisible(Eigen::Ref<const RowMatrix<double>> x) {
  auto& visible = current_X_;
  CheckShape(__FUNCTION__, "v", {x.rows(), x.cols()},
             {visible.rows(), visible.cols()});

  visible = x;
}

void MetropolisHastings::OneStep() {
  transition_kernel_(
      current_X_, proposed_X_,
      log_acceptance_correction_);  // Now proposed_X_ contains next states `v'`

  LogValBatched(proposed_X_, /*out=*/proposed_Y_, /*cache=*/{});

  // Calculates acceptance probability
  quotient_Y_ = (proposed_Y_ - current_Y_ + log_acceptance_correction_).exp();

  GetMachineFunc()(quotient_Y_, probability_);

  for (auto i = Index{0}; i < accept_.size(); ++i) {
    accept_(i) = probability_(i) >= 1.0
                     ? true
                     : std::uniform_real_distribution<double>{}(
                           GetRandomEngine()) < probability_(i);

    // Updates current state
    if (accept_(i)) {
      current_X_.row(i) = proposed_X_.row(i);
      accepted_samples_ += 1;
    }
  }

  current_Y_ = accept_.select(proposed_Y_, current_Y_);

  // Update acceptance counters
  total_samples_ += accept_.size();
}

Index MetropolisHastings::BatchSize() const noexcept { return n_chains_; }

Index MetropolisHastings::NChains() const noexcept { return n_chains_; }

Index MetropolisHastings::SweepSize() const noexcept { return sweep_size_; }

void MetropolisHastings::SweepSize(Index const sweep_size) {
  detail::CheckSweepSize(__FUNCTION__, sweep_size);
  sweep_size_ = sweep_size;
}

void MetropolisHastings::Sweep() {
  for (auto i = Index{0}; i < sweep_size_; ++i) {
    OneStep();
  }
}

// Creates sub-batched calls of LogVal
void MetropolisHastings::LogValBatched(
    Eigen::Ref<const RowMatrix<double>> v,
    Eigen::Ref<AbstractMachine::VectorType> out, const any& cache) {
  Index imax = v.rows() / batch_size_;

  for (Index i = 0; i < imax; i++) {
    GetMachine().LogVal(v.block(i * batch_size_, 0, batch_size_, v.cols()),
                        out.block(i * batch_size_, 0, batch_size_, out.cols()),
                        cache);
  }

  if (batch_size_ * imax < v.rows()) {
    GetMachine().LogVal(v.bottomRows(v.rows() - batch_size_ * imax),
                        out.bottomRows(v.rows() - batch_size_ * imax), cache);
  }
}

}  // namespace netket
