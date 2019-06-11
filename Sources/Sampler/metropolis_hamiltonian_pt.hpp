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

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

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

  int mynode_;
  int totalnodes_;

  // Look-up tables
  std::vector<typename AbstractMachine::LookupType> lt_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  std::vector<std::vector<int>> tochange1_;
  std::vector<std::vector<double>> newconfs1_;
  std::vector<Complex> mel1_;

 public:
  MetropolisHamiltonianPt(AbstractMachine &psi, H &hamiltonian, int nrep)
      : AbstractSampler(psi),
        hamiltonian_(hamiltonian),
        nv_(GetHilbert().Size()),
        nrep_(nrep) {
    Init();
  }

  void Init() {
    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (!GetHilbert().IsDiscrete()) {
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

    InfoMessage() << "Hamiltonian Metropolis sampler with parallel tempering "
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
    for (int i = 0; i < nv_; i++) {
      hamiltonian_.FindConn(v_[rep], mel_, tochange_, newconfs_);

      const double w1 = tochange_.size();

      std::uniform_int_distribution<int> distrs(0, tochange_.size() - 1);
      std::uniform_real_distribution<double> distu(0, 1);

      // picking a random state to transit to
      int si = distrs(this->GetRandomEngine());

      // Inverse transition
      v1_ = v_[rep];
      GetHilbert().UpdateConf(v1_, tochange_[si], newconfs_[si]);

      hamiltonian_.FindConn(v1_, mel1_, tochange1_, newconfs1_);

      double w2 = tochange1_.size();

      const auto lvd =
          GetMachine().LogValDiff(v_[rep], tochange_[si], newconfs_[si], lt_[rep]);
      double ratio =
          this->GetMachineFunc()(std::exp(beta_[rep] * lvd)) * w1 / w2;

#ifndef NDEBUG
      const auto psival1 = GetMachine().LogVal(v_[rep]);
      if (std::abs(
              std::exp(GetMachine().LogVal(v_[rep]) - GetMachine().LogVal(v_[rep], lt_[rep])) -
              1.) > 1.0e-8) {
        std::cerr << GetMachine().LogVal(v_[rep]) << "  and LogVal with Lt is "
                  << GetMachine().LogVal(v_[rep], lt_[rep]) << std::endl;
        std::abort();
      }
#endif

      // Metropolis acceptance test
      if (ratio > distu(this->GetRandomEngine())) {
        accept_(rep) += 1;
        GetMachine().UpdateLookup(v_[rep], tochange_[si], newconfs_[si], lt_[rep]);
        v_[rep] = v1_;

#ifndef NDEBUG
        const auto psival2 = GetMachine().LogVal(v_[rep]);
        if (std::abs(std::exp(psival2 - psival1 - lvd) - 1.) > 1.0e-8) {
          std::cerr << psival2 - psival1 << " and logvaldiff is " << lvd
                    << std::endl;
          std::cerr << psival2 << " and LogVal with Lt is "
                    << GetMachine().LogVal(v_[rep], lt_[rep]) << std::endl;
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
    const double lf1 = 2 * std::real(GetMachine().LogVal(v_[r1], lt_[r1]));
    const double lf2 = 2 * std::real(GetMachine().LogVal(v_[r2], lt_[r2]));

    return std::exp((beta_[r1] - beta_[r2]) * (lf2 - lf1));
  }

  void Exchange(int r1, int r2) {
    std::swap(v_[r1], v_[r2]);
    std::swap(lt_[r1], lt_[r2]);
  }

  Eigen::VectorXd Visible() override { return v_[0]; }

  void SetVisible(const Eigen::VectorXd &v) override { v_[0] = v; }

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
