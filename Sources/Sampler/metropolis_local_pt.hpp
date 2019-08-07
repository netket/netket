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

#ifndef NETKET_METROPOLISFLIPT_HPP
#define NETKET_METROPOLISFLIPT_HPP

#include <Eigen/Core>
#include "Sampler/abstract_sampler.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Metropolis sampling generating local changes
// Parallel tempering is also used
class MetropolisLocalPt : public AbstractSampler {
  // number of visible units
  const int nv_;

  // states of visible units
  // for each sampled temperature
  std::vector<Eigen::VectorXd> v_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  // clusters to do updates
  std::vector<std::vector<int>> clusters_;

  // Look-up tables
  std::vector<any> lt_;

  int nrep_;

  std::vector<double> beta_;

  int nstates_;
  std::vector<double> localstates_;

  int sweep_size_;

  LogValAccumulator log_val_accumulator_;

 public:
  // Constructor with one replica by default
  explicit MetropolisLocalPt(AbstractMachine& psi, int nreplicas = 1)
      : AbstractSampler(psi),
        nv_(GetMachine().GetHilbert().Size()),
        nrep_(nreplicas) {
    Init();
  }

  void Init() {
    nstates_ = GetMachine().GetHilbert().LocalSize();
    localstates_ = GetMachine().GetHilbert().LocalStates();

    SetNreplicas(nrep_);

    // Always use odd sweep size to avoid possible ergodicity problems
    if (nv_ % 2 == 0) {
      sweep_size_ = nv_ + 1;
    } else {
      sweep_size_ = nv_;
    }

    InfoMessage() << "Metropolis sampler with parallel tempering is ready "
                  << std::endl;
    InfoMessage() << "Nreplicas is equal to " << nrep_ << std::endl;
  }

  void SetNreplicas(int nrep) {
    nrep_ = nrep;
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
  }

  void Reset(bool initrandom = false) override {
    if (initrandom) {
      for (int i = 0; i < nrep_; i++) {
        GetMachine().GetHilbert().RandomVals(v_[i], GetRandomEngine());
      }
    }

    for (int i = 0; i < nrep_; i++) {
      lt_[i] = GetMachine().InitLookup(v_[i]);
    }
    log_val_accumulator_ = GetMachine().LogValSingle(v_[0], lt_[0]);

    accept_ = Eigen::VectorXd::Zero(2 * nrep_);
    moves_ = Eigen::VectorXd::Zero(2 * nrep_);
  }

  // Exchange sweep at given temperature
  void LocalSweep(int rep) {
    std::vector<int> tochange(1);
    std::vector<double> newconf(1);

    std::uniform_real_distribution<double> distu;
    std::uniform_int_distribution<int> distrs(0, nv_ - 1);
    std::uniform_int_distribution<int> diststate(0, nstates_ - 1);

    for (int i = 0; i < sweep_size_; i++) {
      // picking a random site to be changed
      int si = distrs(GetRandomEngine());
      assert(si < nv_);
      tochange[0] = si;

      // picking a random state
      int newstate = diststate(GetRandomEngine());
      newconf[0] = localstates_[newstate];

      // make sure that the new state is not equal to the current one
      while (std::abs(newconf[0] - v_[rep](si)) <
             std::numeric_limits<double>::epsilon()) {
        newstate = diststate(GetRandomEngine());
        newconf[0] = localstates_[newstate];
      }

      const auto lvd =
          GetMachine().LogValDiff(v_[rep], tochange, newconf, lt_[rep]);
      double ratio =
          NETKET_SAMPLER_APPLY_MACHINE_FUNC(std::exp(beta_[rep] * lvd));

#ifndef NDEBUG
      const auto psival1 = GetMachine().LogValSingle(v_[rep]);
      if (std::abs(std::exp(GetMachine().LogValSingle(v_[rep]) -
                            GetMachine().LogValSingle(v_[rep], lt_[rep])) -
                   1.) > 1.0e-8) {
        std::cerr << GetMachine().LogValSingle(v_[rep])
                  << "  and LogVal with Lt is "
                  << GetMachine().LogValSingle(v_[rep], lt_[rep]) << std::endl;
        std::abort();
      }
#endif
      // Metropolis acceptance test
      if (ratio > distu(GetRandomEngine())) {
        accept_(rep) += 1;

        GetMachine().UpdateLookup(v_[rep], tochange, newconf, lt_[rep]);
        GetMachine().GetHilbert().UpdateConf(v_[rep], tochange, newconf);
        if (rep == 0) {
          log_val_accumulator_ += lvd;
        }

#ifndef NDEBUG
        const auto psival2 = GetMachine().LogValSingle(v_[rep]);
        if (std::abs(std::exp(psival2 - psival1 - lvd) - 1.) > 1.0e-8) {
          std::cerr << psival2 - psival1 << " and logvaldiff is " << lvd
                    << std::endl;
          std::cerr << psival2 << " and LogVal with Lt is "
                    << GetMachine().LogValSingle(v_[rep], lt_[rep])
                    << std::endl;
          std::abort();
        }
#endif
      }
      moves_(rep) += 1;
    }
  }

  void Sweep() override {
    // First we do local sweeps
    for (int i = 0; i < nrep_; i++) {
      LocalSweep(i);
    }

    // Tempearture exchanges
    std::uniform_real_distribution<double> distribution(0, 1);

    for (int r = 1; r < nrep_; r += 2) {
      if (ExchangeProb(r, r - 1) > distribution(GetRandomEngine())) {
        Exchange(r, r - 1);
        accept_(nrep_ + r) += 1.;
        accept_(nrep_ + r - 1) += 1;
      }
      moves_(nrep_ + r) += 1.;
      moves_(nrep_ + r - 1) += 1;
    }

    for (int r = 2; r < nrep_; r += 2) {
      if (ExchangeProb(r, r - 1) > distribution(GetRandomEngine())) {
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
    const auto lf1 = GetMachine().LogValSingle(v_[r1], lt_[r1]);
    const auto lf2 = GetMachine().LogValSingle(v_[r2], lt_[r2]);

    return NETKET_SAMPLER_APPLY_MACHINE_FUNC(
        std::exp((beta_[r1] - beta_[r2]) * (lf2 - lf1)));
  }

  void Exchange(int r1, int r2) {
    std::swap(v_[r1], v_[r2]);
    std::swap(lt_[r1], lt_[r2]);
    if (r1 == 0 || r2 == 0) {
      log_val_accumulator_ = GetMachine().LogValSingle(v_[0], lt_[0]);
    }
  }

  std::pair<Eigen::Ref<const RowMatrix<double>>,
            Eigen::Ref<const Eigen::VectorXcd>>
  CurrentState() const override {
    return {v_[0].transpose(), Eigen::Map<const Eigen::VectorXcd>{
                                   &log_val_accumulator_.LogVal(), 1}};
  }

  NETKET_SAMPLER_SET_VISIBLE_DEFAULT(v_[0])
  NETKET_SAMPLER_ACCEPTANCE_DEFAULT_PT(accept_, moves_)

  Index BatchSize() const noexcept override { return 1; }
};

}  // namespace netket

#endif
