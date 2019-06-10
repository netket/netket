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

// authors: Hugo Théveniaut and Fabien Alet

#ifndef NETKET_CUSTOMSAMPLERPT_HPP
#define NETKET_CUSTOMSAMPLERPT_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include "Operator/local_operator.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

namespace netket {

// Metropolis sampling using custom moves provided by user
class CustomSamplerPt : public AbstractSampler {
  LocalOperator move_operators_;
  std::vector<double> operatorsweights_;
  // number of visible units
  const int nv_;

  // states of visible units
  // for each sampled temperature
  std::vector<Eigen::VectorXd> v_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  // Look-up tables
  std::vector<typename AbstractMachine::LookupType> lt_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  int nstates_;
  std::vector<double> localstates_;

  int nrep_;
  std::vector<double> beta_;

 public:
  CustomSamplerPt(AbstractMachine& psi, const LocalOperator& move_operators,
                  const std::vector<double>& move_weights = {},
                  int nreplicas = 1)
      : AbstractSampler(psi),
        move_operators_(move_operators),
        nv_(GetHilbert().Size()),
        nrep_(nreplicas) {
    Init(move_weights);
  }

  void Init(const std::vector<double>& move_weights) {
    CustomSampler::CheckMoveOperators(move_operators_);

    if (GetHilbert().Size() != move_operators_.GetHilbert().Size()) {
      throw InvalidInputError(
          "Move operators in CustomSampler act on a different hilbert space "
          "than the Machine");
    }

    if (move_weights.size()) {
      operatorsweights_ = move_weights;

      if (operatorsweights_.size() != move_operators_.Size()) {
        throw InvalidInputError(
            "The custom sampler definition is inconsistent (between "
            "MoveWeights and MoveOperators sizes)");
      }

    } else {  // By default the stochastic operators are drawn uniformly
      operatorsweights_.resize(move_operators_.Size(), 1.0);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (!GetHilbert().IsDiscrete()) {
      throw InvalidInputError(
          "Custom Metropolis sampler works only for discrete Hilbert spaces");
    }

    nstates_ = GetHilbert().LocalSize();
    localstates_ = GetHilbert().LocalStates();

    v_.resize(nrep_);
    for (int i = 0; i < nrep_; i++) {
      v_[i].resize(nv_);
    }

    for (int i = 0; i < nrep_; i++) {
      beta_.push_back(1. - double(i) / double(nrep_));
    }

    lt_.resize(nrep_);

    accept_.resize(2 * nrep_);
    moves_.resize(2 * nrep_);

    Reset(true);

    InfoMessage() << "Custom Metropolis sampler with parallel tempering "
                     "is ready "
                  << std::endl;
    InfoMessage() << nrep_ << " replicas are being used" << std::endl;
  }

  void Reset(bool initrandom = false) override {
    if (initrandom) {
      for (int i = 0; i < nrep_; i++) {
        GetHilbert().RandomVals(v_[i], this->GetRandomEngine());
      }
    }

    for (int i = 0; i < nrep_; i++) {
      GetMachine().InitLookup(v_[i], lt_[i]);
    }

    accept_ = Eigen::VectorXd::Zero(2 * nrep_);
    moves_ = Eigen::VectorXd::Zero(2 * nrep_);
  }

  void LocalSweep(int rep) {
    std::discrete_distribution<int> disc_dist(operatorsweights_.begin(),
                                              operatorsweights_.end());
    std::uniform_real_distribution<double> distu;
    for (int i = 0; i < nv_; i++) {
      // pick a random operator in possible ones according to the provided
      // weights
      int op = disc_dist(this->GetRandomEngine());
      move_operators_.FindConn(op, v_[rep], mel_, tochange_, newconfs_);

      double p = distu(this->GetRandomEngine());
      std::size_t exit_state = 0;
      double cumulative_prob = std::real(mel_[0]);
      while (p > cumulative_prob) {
        exit_state++;
        cumulative_prob += std::real(mel_[exit_state]);
      }

      auto explo = std::exp(beta_[rep] *
                            GetMachine().LogValDiff(v_[rep], tochange_[exit_state],
                                            newconfs_[exit_state], lt_[rep]));

      double ratio = this->GetMachineFunc()(explo);

      // Metropolis acceptance test
      if (ratio > distu(this->GetRandomEngine())) {
        accept_(rep) += 1;
        GetMachine().UpdateLookup(v_[rep], tochange_[exit_state], newconfs_[exit_state],
                          lt_[rep]);
        GetHilbert().UpdateConf(v_[rep], tochange_[exit_state],
                            newconfs_[exit_state]);
      }
      moves_(rep) += 1;
    }
  }

  void Sweep() override {
    // First we do local sweeps
    for (int i = 0; i < nrep_; i++) {
      LocalSweep(i);
    }

    // Temperature exchanges
    std::uniform_real_distribution<double> distribution(0, 1);

    for (int r = 1; r < nrep_; r += 2) {
      if (ExchangeProb(r, r - 1) > distribution(this->GetRandomEngine())) {
        Exchange(r, r - 1);
        accept_(nrep_ + r) += 1.;
        accept_(nrep_ + r - 1) += 1;
      }
      moves_(nrep_ + r) += 1.;
      moves_(nrep_ + r - 1) += 1;
    }

    for (int r = 2; r < nrep_; r += 2) {
      if (ExchangeProb(r, r - 1) > distribution(this->GetRandomEngine())) {
        Exchange(r, r - 1);
        accept_(nrep_ + r) += 1.;
        accept_(nrep_ + r - 1) += 1;
      }
      moves_(nrep_ + r) += 1.;
      moves_(nrep_ + r - 1) += 1;
    }
  }

  // computes the probability to exchange two replicas
  double ExchangeProb(int r1, int r2) {
    const double lf1 = 2 * std::real(GetMachine().LogVal(v_[r1], lt_[r1]));
    const double lf2 = 2 * std::real(GetMachine().LogVal(v_[r2], lt_[r2]));

    return std::exp((beta_[r1] - beta_[r2]) * (lf2 - lf1));
  }

  void Exchange(int r1, int r2) {
    std::swap(v_[r1], v_[r2]);
    std::swap(lt_[r1], lt_[r2]);
  }

  Eigen::VectorXd Visible() override { return v_[0]; }

  void SetVisible(const Eigen::VectorXd& v) override { v_[0] = v; }

  AbstractMachine::VectorType DerLogVisible() override {
    return GetMachine().DerLog(v_[0], lt_[0]);
  }

  Eigen::VectorXd Acceptance() const override {
    Eigen::VectorXd acc = accept_;
    for (int i = 0; i < acc.size(); i++) {
      acc(i) /= moves_(i);
    }
    return acc;
  }
};
}  // namespace netket

#endif
