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

#ifndef NETKET_METROPOLISHAMILTONIAN_PT_HPP
#define NETKET_METROPOLISHAMILTONIAN_PT_HPP

#include <Eigen/Core>
#include "Sampler/abstract_sampler.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Metropolis sampling generating transitions using the Hamiltonian
template <class H>
class MetropolisHamiltonianPt : public AbstractSampler {
  H &hamiltonian_;

  // number of visible units
  const int nv_;

  const int nrep_;
  std::vector<double> beta_;

  // states of visible units
  std::vector<Eigen::VectorXd> v_;
  Eigen::VectorXd v1_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  // Look-up tables
  std::vector<any> lt_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  std::vector<std::vector<int>> tochange1_;
  std::vector<std::vector<double>> newconfs1_;
  std::vector<Complex> mel1_;

  int sweep_size_;

  LogValAccumulator log_val_accumulator_;

 public:
  MetropolisHamiltonianPt(AbstractMachine &psi, H &hamiltonian, int nrep)
      : AbstractSampler(psi),
        hamiltonian_(hamiltonian),
        nv_(GetMachine().GetHilbert().Size()),
        nrep_(nrep) {
    Init();
  }

  void Init() {
    if (!GetMachine().GetHilbert().IsDiscrete()) {
      throw InvalidInputError(
          "Hamiltonian Metropolis sampler works only for discrete "
          "Hilbert spaces");
    }

    v_.resize(nrep_);
    for (int i = 0; i < nrep_; i++) {
      v_[i].resize(nv_);
    }

    for (int i = 0; i < nrep_; i++) {
      beta_.push_back(1. - double(i) / double(nrep_));
    }

    accept_.resize(2 * nrep_);
    moves_.resize(2 * nrep_);

    lt_.resize(nrep_);

    Reset(true);

    // Always use odd sweep size to avoid possible ergodicity problems
    if (nv_ % 2 == 0) {
      sweep_size_ = nv_ + 1;
    } else {
      sweep_size_ = nv_;
    }

    InfoMessage() << "Hamiltonian Metropolis sampler with parallel tempering "
                     "is ready "
                  << std::endl;
    InfoMessage() << nrep_ << " replicas are being used" << std::endl;
  }

  void Reset(bool initrandom = false) override {
    if (initrandom) {
      for (int i = 0; i < nrep_; i++) {
        GetMachine().GetHilbert().RandomVals(v_[i], this->GetRandomEngine());
      }
    }

    for (int i = 0; i < nrep_; i++) {
      lt_[i] = GetMachine().InitLookup(v_[i]);
    }
    log_val_accumulator_ = GetMachine().LogValSingle(v_[0], lt_[0]);

    accept_ = Eigen::VectorXd::Zero(2 * nrep_);
    moves_ = Eigen::VectorXd::Zero(2 * nrep_);
  }

  void LocalSweep(int rep) {
    for (int i = 0; i < sweep_size_; i++) {
      hamiltonian_.FindConn(v_[rep], mel_, tochange_, newconfs_);

      const double w1 = tochange_.size();

      std::uniform_int_distribution<int> distrs(0, tochange_.size() - 1);
      std::uniform_real_distribution<double> distu(0, 1);

      // picking a random state to transit to
      int si = distrs(this->GetRandomEngine());

      // Inverse transition
      v1_ = v_[rep];
      GetMachine().GetHilbert().UpdateConf(v1_, tochange_[si], newconfs_[si]);

      hamiltonian_.FindConn(v1_, mel1_, tochange1_, newconfs1_);

      double w2 = tochange1_.size();

      const auto lvd = GetMachine().LogValDiff(v_[rep], tochange_[si],
                                               newconfs_[si], lt_[rep]);
      double ratio =
          NETKET_SAMPLER_APPLY_MACHINE_FUNC(std::exp(beta_[rep] * lvd)) * w1 /
          w2;

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
      if (ratio > distu(this->GetRandomEngine())) {
        accept_(rep) += 1;
        GetMachine().UpdateLookup(v_[rep], tochange_[si], newconfs_[si],
                                  lt_[rep]);
        v_[rep] = v1_;
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
    // First we do local exchange sweeps
    for (int i = 0; i < nrep_; i++) {
      LocalSweep(i);
    }

    // Tempearture exchanges
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
