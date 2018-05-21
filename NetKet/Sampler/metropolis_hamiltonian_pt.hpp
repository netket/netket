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
template <class WfType, class H>
class MetropolisHamiltonianPt : public AbstractSampler<WfType> {
  WfType &psi_;

  const Hilbert &hilbert_;

  H &hamiltonian_;

  // number of visible units
  const int nv_;

  netket::default_random_engine rgen_;

  // states of visible units
  std::vector<Eigen::VectorXd> v_;
  Eigen::VectorXd v1_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  // Look-up tables
  std::vector<typename WfType::LookupType> lt_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<std::complex<double>> mel_;

  std::vector<std::vector<int>> tochange1_;
  std::vector<std::vector<double>> newconfs1_;
  std::vector<std::complex<double>> mel1_;

  const int nrep_;
  std::vector<double> beta_;

 public:
  MetropolisHamiltonianPt(WfType &psi, H &hamiltonian, int nrep)
      : psi_(psi),
        hilbert_(psi.GetHilbert()),
        nv_(hilbert_.Size()),
        hamiltonian_(hamiltonian),
        nrep_(nrep) {
    Init();
  }

  // Json constructor
  MetropolisHamiltonianPt(WfType &psi, H &hamiltonian, const json &pars)
      : psi_(psi),
        hilbert_(psi.GetHilbert()),
        hamiltonian_(hamiltonian),
        nv_(hilbert_.Size()),
        nrep_(FieldVal(pars["Sampler"], "Nreplicas")) {
    Init();
  }

  void Init() {
    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (!hilbert_.IsDiscrete()) {
      if (mynode_ == 0) {
        std::cerr << "# Hamiltonian Metropolis sampler works only for discrete "
                     "Hilbert spaces"
                  << std::endl;
      }
      std::abort();
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

    Seed();

    Reset(true);

    if (mynode_ == 0) {
      std::cout << "# Hamiltonian Metropolis sampler with parallel tempering "
                   "is ready "
                << std::endl;
      std::cout << "# " << nrep_ << " replicas are being used" << std::endl;
    }
  }

  void Seed(int baseseed = 0) {
    std::random_device rd;
    std::vector<int> seeds(totalnodes_);

    if (mynode_ == 0) {
      for (int i = 0; i < totalnodes_; i++) {
        seeds[i] = rd() + baseseed;
      }
    }

    SendToAll(seeds);

    rgen_.seed(seeds[mynode_]);
  }

  void Reset(bool initrandom = false) override {
    if (initrandom) {
      for (int i = 0; i < nrep_; i++) {
        hilbert_.RandomVals(v_[i], rgen_);
      }
    }

    for (int i = 0; i < nrep_; i++) {
      psi_.InitLookup(v_[i], lt_[i]);
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
      int si = distrs(rgen_);

      // Inverse transition
      v1_ = v_[rep];
      hilbert_.UpdateConf(v1_, tochange_[si], newconfs_[si]);

      hamiltonian_.FindConn(v1_, mel1_, tochange1_, newconfs1_);

      double w2 = tochange1_.size();

      const auto lvd =
          psi_.LogValDiff(v_[rep], tochange_[si], newconfs_[si], lt_[rep]);
      double ratio = std::norm(std::exp(beta_[rep] * lvd) * w1 / w2);

#ifndef NDEBUG
      const auto psival1 = psi_.LogVal(v_[rep]);
      if (std::abs(
              std::exp(psi_.LogVal(v_[rep]) - psi_.LogVal(v_[rep], lt_[rep])) -
              1.) > 1.0e-8) {
        std::cerr << psi_.LogVal(v_[rep]) << "  and LogVal with Lt is "
                  << psi_.LogVal(v_[rep], lt_[rep]) << std::endl;
        std::abort();
      }
#endif

      // Metropolis acceptance test
      if (ratio > distu(rgen_)) {
        accept_(rep) += 1;
        psi_.UpdateLookup(v_[rep], tochange_[si], newconfs_[si], lt_[rep]);
        v_[rep] = v1_;

#ifndef NDEBUG
        const auto psival2 = psi_.LogVal(v_[rep]);
        if (std::abs(std::exp(psival2 - psival1 - lvd) - 1.) > 1.0e-8) {
          std::cerr << psival2 - psival1 << " and logvaldiff is " << lvd
                    << std::endl;
          std::cerr << psival2 << " and LogVal with Lt is "
                    << psi_.LogVal(v_[rep], lt_[rep]) << std::endl;
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
      if (ExchangeProb(r, r - 1) > distribution(rgen_)) {
        Exchange(r, r - 1);
        accept_(nrep_ + r) += 1.;
        accept_(nrep_ + r - 1) += 1;
      }
      moves_(nrep_ + r) += 1.;
      moves_(nrep_ + r - 1) += 1;
    }

    for (int r = 2; r < nrep_; r += 2) {
      if (ExchangeProb(r, r - 1) > distribution(rgen_)) {
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
    const double lf1 = 2 * std::real(psi_.LogVal(v_[r1], lt_[r1]));
    const double lf2 = 2 * std::real(psi_.LogVal(v_[r2], lt_[r2]));

    return std::exp((beta_[r1] - beta_[r2]) * (lf2 - lf1));
  }

  void Exchange(int r1, int r2) {
    std::swap(v_[r1], v_[r2]);
    std::swap(lt_[r1], lt_[r2]);
  }

  Eigen::VectorXd Visible() override { return v_[0]; }

  void SetVisible(const Eigen::VectorXd &v) override { v_[0] = v; }

  WfType &Psi() override { return psi_; }

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
