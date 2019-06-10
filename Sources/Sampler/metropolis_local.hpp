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

#ifndef NETKET_METROPOLISLOCAL_HPP
#define NETKET_METROPOLISLOCAL_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

namespace netket {

// Metropolis sampling generating local moves in hilbert space
class MetropolisLocal : public AbstractSampler {
  // number of visible units
  const int nv_;

  // states of visible units
  Eigen::VectorXd v_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  // Look-up tables
  typename AbstractMachine::LookupType lt_;

  int nstates_;
  std::vector<double> localstates_;

 public:
  explicit MetropolisLocal(AbstractMachine& psi)
      : AbstractSampler(psi), nv_(GetHilbert().Size()) {
    Init();
  }

  void Init() {
    v_.resize(nv_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (!GetHilbert().IsDiscrete()) {
      throw InvalidInputError(
          "Local Metropolis sampler works only for discrete "
          "Hilbert spaces");
    }

    accept_.resize(1);
    moves_.resize(1);

    nstates_ = GetHilbert().LocalSize();
    localstates_ = GetHilbert().LocalStates();

    Reset(true);

    InfoMessage() << "Local Metropolis sampler is ready " << std::endl;
  }

  void Reset(bool initrandom) override {
    if (initrandom) {
      GetHilbert().RandomVals(v_, this->GetRandomEngine());
    }

    GetMachine().InitLookup(v_, lt_);

    accept_ = Eigen::VectorXd::Zero(1);
    moves_ = Eigen::VectorXd::Zero(1);
  }

  void Sweep() override {
    std::vector<int> tochange(1);
    std::vector<double> newconf(1);

    std::uniform_real_distribution<double> distu;
    std::uniform_int_distribution<int> distrs(0, nv_ - 1);
    std::uniform_int_distribution<int> diststate(0, nstates_ - 1);

    for (int i = 0; i < nv_; i++) {
      // picking a random site to be changed
      int si = distrs(this->GetRandomEngine());
      assert(si < nv_);
      tochange[0] = si;

      // picking a random state
      int newstate = diststate(this->GetRandomEngine());
      newconf[0] = localstates_[newstate];

      // make sure that the new state is not equal to the current one
      while (std::abs(newconf[0] - v_(si)) <
             std::numeric_limits<double>::epsilon()) {
        newstate = diststate(this->GetRandomEngine());
        newconf[0] = localstates_[newstate];
      }

      const auto lvd = GetMachine().LogValDiff(v_, tochange, newconf, lt_);
      double ratio = this->GetMachineFunc()(std::exp(lvd));

#ifndef NDEBUG
      const auto psival1 = GetMachine().LogVal(v_);
      if (std::abs(std::exp(GetMachine().LogVal(v_) - GetMachine().LogVal(v_, lt_)) - 1.) >
          1.0e-8) {
        std::cerr << GetMachine().LogVal(v_) << "  and LogVal with Lt is "
                  << GetMachine().LogVal(v_, lt_) << std::endl;
        std::abort();
      }
#endif

      // Metropolis acceptance test
      if (ratio > distu(this->GetRandomEngine())) {
        accept_[0] += 1;
        GetMachine().UpdateLookup(v_, tochange, newconf, lt_);
        GetHilbert().UpdateConf(v_, tochange, newconf);

#ifndef NDEBUG
        const auto psival2 = GetMachine().LogVal(v_);
        if (std::abs(std::exp(psival2 - psival1 - lvd) - 1.) > 1.0e-8) {
          std::cerr << psival2 - psival1 << " and logvaldiff is " << lvd
                    << std::endl;
          std::cerr << psival2 << " and LogVal with Lt is "
                    << GetMachine().LogVal(v_, lt_) << std::endl;
          std::abort();
        }
#endif
      }
      moves_[0] += 1;
    }
  }

  Eigen::VectorXd Visible() override { return v_; }

  void SetVisible(const Eigen::VectorXd& v) override { v_ = v; }

  AbstractMachine::VectorType DerLogVisible() override {
    return GetMachine().DerLog(v_, lt_);
  }

  Eigen::VectorXd Acceptance() const override {
    Eigen::VectorXd acc = accept_;
    for (int i = 0; i < 1; i++) {
      acc(i) /= moves_(i);
    }
    return acc;
  }
};

}  // namespace netket

#endif
