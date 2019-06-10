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

#ifndef NETKET_METROPOLISEXCHANGE_HPP
#define NETKET_METROPOLISEXCHANGE_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

namespace netket {

// Metropolis sampling generating local exchanges
class MetropolisExchange : public AbstractSampler {
  // number of visible units
  const int nv_;

  // states of visible units
  Eigen::VectorXd v_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  // clusters to do updates
  std::vector<std::vector<int>> clusters_;

  // Look-up tables
  typename AbstractMachine::LookupType lt_;

 public:
  MetropolisExchange(const AbstractGraph &graph, AbstractMachine &psi,
                     int dmax = 1)
      : AbstractSampler(psi), nv_(GetHilbert().Size()) {
    Init(graph, dmax);
  }

  template <class G>
  void Init(const G &graph, int dmax) {
    v_.resize(nv_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    accept_.resize(1);
    moves_.resize(1);

    GenerateClusters(graph, dmax);

    Reset(true);

    InfoMessage() << "Metropolis Exchange sampler is ready " << std::endl;
    InfoMessage() << dmax << " is the maximum distance for exchanges"
                  << std::endl;
  }

  template <class G>
  void GenerateClusters(const G &graph, int dmax) {
    auto dist = graph.AllDistances();

    assert(int(dist.size()) == nv_);

    for (int i = 0; i < nv_; i++) {
      for (int j = 0; j < nv_; j++) {
        if (dist[i][j] <= dmax && i != j) {
          clusters_.push_back({i, j});
        }
      }
    }
  }

  void Reset(bool initrandom = false) override {
    if (initrandom) {
      if (initrandom) {
        GetHilbert().RandomVals(v_, this->GetRandomEngine());
      }
    }

    GetMachine().InitLookup(v_, lt_);

    accept_ = Eigen::VectorXd::Zero(1);
    moves_ = Eigen::VectorXd::Zero(1);
  }

  void Sweep() override {
    std::vector<int> tochange(2);
    std::uniform_real_distribution<double> distu;
    std::uniform_int_distribution<int> distcl(0, clusters_.size() - 1);

    std::vector<double> newconf(2);

    for (int i = 0; i < nv_; i++) {
      int rcl = distcl(this->GetRandomEngine());
      assert(rcl < int(clusters_.size()));
      int si = clusters_[rcl][0];
      int sj = clusters_[rcl][1];

      assert(si < nv_ && sj < nv_);

      if (std::abs(v_(si) - v_(sj)) > std::numeric_limits<double>::epsilon()) {
        tochange = clusters_[rcl];
        newconf[0] = v_(sj);
        newconf[1] = v_(si);

        auto explo = std::exp(GetMachine().LogValDiff(v_, tochange, newconf, lt_));

        double ratio = this->GetMachineFunc()(explo);

        if (ratio > distu(this->GetRandomEngine())) {
          accept_[0] += 1;
          GetMachine().UpdateLookup(v_, tochange, newconf, lt_);
          GetHilbert().UpdateConf(v_, tochange, newconf);
        }
      }
      moves_[0] += 1;
    }
  }

  Eigen::VectorXd Visible() override { return v_; }

  void SetVisible(const Eigen::VectorXd &v) override { v_ = v; }

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
