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

#include "Sampler/metropolis_hastings_pt.hpp"

namespace netket {

MetropolisHastingsPt::MetropolisHastingsPt(
    AbstractMachine& machine,
    MetropolisHastingsPt::TransitionKernel transition_kernel, Index n_replicas,
    Index sweep_size)
    : AbstractSampler(machine),
      transition_kernel_(transition_kernel),
      n_replicas_(n_replicas),
      sweep_size_(sweep_size) {
  detail::CheckReplicasSize(__FUNCTION__, n_replicas);
  detail::CheckSweepSize(__FUNCTION__, sweep_size);

  current_X_.resize(n_replicas, machine.Nvisible());
  proposed_X_.resize(n_replicas, machine.Nvisible());
  current_Y_.resize(n_replicas);
  proposed_Y_.resize(n_replicas);
  quotient_Y_.resize(n_replicas);
  probability_.resize(n_replicas);
  accept_.resize(n_replicas);
  log_acceptance_correction_.resize(n_replicas);

  beta_.resize(n_replicas);
  proposed_beta_.resize(n_replicas);

  accepted_samples_.resize(n_replicas);

  // Linearly spaced inverse temperature
  for (Index i = 0; i < n_replicas; i++) {
    beta_(i) = (1. - double(i) / double(n_replicas));
  }

  beta1_ind_ = 0;

  Reset(true);
}

void MetropolisHastingsPt::Reset(bool init_random) {
  if (init_random) {
    for (Index i = 0; i < n_replicas_; i++) {
      GetMachine().GetHilbert().RandomVals(current_X_.row(i),
                                           GetRandomEngine());
    }
  }
  GetMachine().LogVal(current_X_, current_Y_, {});

  accepted_samples_.setZero();
  total_samples_ = 0;
  total_exchange_steps_ = 0;
  beta1_av_ = 0;
  beta1_av_sq_ = 0;
}

std::pair<Eigen::Ref<const RowMatrix<double>>,
          Eigen::Ref<const Eigen::VectorXcd>>
MetropolisHastingsPt::CurrentState() const {
  return {Eigen::Map<const RowMatrix<double>>{current_X_.row(beta1_ind_).data(),
                                              1, current_X_.cols()},
          Eigen::Map<const Eigen::VectorXcd>{&current_Y_(beta1_ind_), 1}};
}

void MetropolisHastingsPt::SetVisible(Eigen::Ref<const RowMatrix<double>> x) {
  CheckShape(__FUNCTION__, "v", {x.rows(), x.cols()}, {1, current_X_.cols()});

  current_X_.row(beta1_ind_) = x.row(0);
}

void MetropolisHastingsPt::OneStep() {
  transition_kernel_(current_X_, proposed_X_,
                     log_acceptance_correction_);  // Now proposed_X_ contains
                                                   // next states `v'`

  GetMachine().LogVal(proposed_X_, /*out=*/proposed_Y_, /*cache=*/{});

  // Calculates acceptance probability
  quotient_Y_ =
      ((proposed_Y_ - current_Y_ + log_acceptance_correction_) * beta_);

  probability_ = (GetMachinePow() * quotient_Y_.real()).exp();

  for (auto i = Index{0}; i < accept_.size(); ++i) {
    accept_(i) = probability_(i) >= 1.0
                     ? true
                     : std::uniform_real_distribution<double>{}(
                           GetRandomEngine()) < probability_(i);

    // Updates current state
    if (accept_(i)) {
      current_X_.row(i) = proposed_X_.row(i);
      // Update acceptance counters
      accepted_samples_(i) += 1.0;
    }
  }

  current_Y_ = accept_.select(proposed_Y_, current_Y_);

  total_samples_ += 1;
}

void MetropolisHastingsPt::ExchangeStep() {
  // Choose a random swap order (odd/even swap)
  Index swap_order =
      std::uniform_int_distribution<Index>(0, 1)(GetRandomEngine());

  ProposePairwiseSwap(beta_, proposed_beta_, swap_order);

  quotient_Y_ = ((proposed_beta_ - beta_) * current_Y_);
  probability_ = (GetMachinePow() * quotient_Y_.real()).exp();

  for (auto i = Index{swap_order}; i < probability_.size(); i += 2) {
    Index inn = (i + 1) % n_replicas_;

    probability_(i) *= probability_(inn);
    accept_(i) = probability_(i) >= 1.0
                     ? true
                     : std::uniform_real_distribution<double>{}(
                           GetRandomEngine()) < probability_(i);

    // Updates current state
    if (accept_(i)) {
      std::swap(beta_(i), beta_(inn));
      std::swap(accepted_samples_(i), accepted_samples_(inn));

      // Keep track of the position of beta=1
      if (beta1_ind_ == i) {
        beta1_ind_ = inn;
      } else if (beta1_ind_ == inn) {
        beta1_ind_ = i;
      }
    }
  }

  total_exchange_steps_ += 1.0;

  // Update statistics to compute diffusion coefficient of replicas
  double delta = beta1_ind_ - beta1_av_;
  beta1_av_ += delta / total_exchange_steps_;
  double delta2 = beta1_ind_ - beta1_av_;
  beta1_av_sq_ += delta * delta2;
}

Index MetropolisHastingsPt::BatchSize() const noexcept { return n_replicas_; }
Index MetropolisHastingsPt::NChains() const noexcept { return 1; }
Index MetropolisHastingsPt::SweepSize() const noexcept { return sweep_size_; }

void MetropolisHastingsPt::SweepSize(Index const sweep_size) {
  detail::CheckSweepSize(__FUNCTION__, sweep_size);
  sweep_size_ = sweep_size;
}

void MetropolisHastingsPt::Sweep() {
  for (auto i = Index{0}; i < sweep_size_; ++i) {
    OneStep();
    ExchangeStep();
  }
}

void MetropolisHastingsPt::ProposePairwiseSwap(
    Eigen::Ref<const Eigen::ArrayXd> vin, Eigen::Ref<Eigen::ArrayXd> vout,
    int order) {
  assert(order == 0 || order == 1);
  assert(vin.size() == vout.size());
  assert(vin.size() % 2 == 0);

  if (order == 0) {
    for (int i = 0; i < vin.size(); i += 2) {
      vout(i) = vin(i + 1);
      vout(i + 1) = vin(i);
    }
  } else if (order == 1) {
    for (int i = 1; i < vin.size() - 1; i += 2) {
      vout(i) = vin(i + 1);
      vout(i + 1) = vin(i);
    }
    vout(0) = vin(vin.size() - 1);
    vout(vout.size() - 1) = vin(0);
  }
}

std::map<std::string, double> MetropolisHastingsPt::Stats() const {
  NETKET_CHECK(total_samples_ > 0 && total_exchange_steps_ > 0, RuntimeError,
               "Cannot compute sampler stats, because no moves were made");

  std::map<std::string, double> stats;

  Eigen::ArrayXd accept =
      accepted_samples_ / static_cast<double>(total_samples_);

  stats["mean_acceptance"] = accept.mean();
  stats["min_acceptance"] = accept.minCoeff();
  stats["max_acceptance"] = accept.maxCoeff();

  // Average position of beta=1
  // This is normalized and centered around zero
  // In the ideal case the average should be zero
  stats["normalized_beta=1_position"] =
      beta1_av_ / static_cast<double>(n_replicas_ - 1) - 0.5;

  // Average variance on the position of beta=1
  // In the ideal case this quantity should be of order ~ [0.2, 1]
  stats["normalized_beta=1_diffusion"] =
      std::sqrt(beta1_av_sq_ / total_exchange_steps_) /
      static_cast<double>(n_replicas_);
  return stats;
}

}  // namespace netket
