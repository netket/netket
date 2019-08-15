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

#ifndef NETKET_HOP_KERNEL_HPP
#define NETKET_HOP_KERNEL_HPP

#include <Eigen/Core>
#include "Utils/random_utils.hpp"

namespace netket {

// Kernel generating local hoppings
class HopKernel {
  // number of visible units
  const Index nv_;

  // clusters to do updates
  std::vector<std::vector<Index>> clusters_;

  Index nstates_;
  std::vector<double> localstates_;

  std::uniform_int_distribution<Index> distcl_;
  std::uniform_int_distribution<Index> diststate_;

 public:
  HopKernel(const AbstractMachine &psi, Index dmax = 1)
      : nv_(psi.GetHilbert().Size()) {
    Init(psi, psi.GetHilbert().GetGraph(), dmax);
  }

  void Init(const AbstractMachine &psi, const AbstractGraph &graph,
            Index dmax) {
    nstates_ = psi.GetHilbert().LocalSize();
    localstates_ = psi.GetHilbert().LocalStates();

    GenerateClusters(graph, dmax);
  }

  void GenerateClusters(const AbstractGraph &graph, int dmax) {
    auto dist = graph.AllDistances();

    assert(Index(dist.size()) == nv_);

    for (Index i = 0; i < nv_; i++) {
      for (Index j = 0; j < nv_; j++) {
        if (dist[i][j] <= dmax && i != j) {
          clusters_.push_back({i, j});
          clusters_.push_back({j, i});
        }
      }
    }

    distcl_ = std::uniform_int_distribution<Index>(0, clusters_.size() - 1);
    diststate_ = std::uniform_int_distribution<Index>(0, nstates_ - 1);
  }

  void operator()(Eigen::Ref<const RowMatrix<double>> v,
                  Eigen::Ref<RowMatrix<double>> vnew,
                  Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction) {
    vnew = v;

    for (int i = 0; i < v.rows(); i++) {
      Index rcl = distcl_(GetRandomEngine());
      assert(rcl < Index(clusters_.size()));
      Index si = clusters_[rcl][0];
      Index sj = clusters_[rcl][1];

      assert(si < nv_ && sj < nv_);

      vnew(i, si) = localstates_[diststate_(GetRandomEngine())];
      vnew(i, sj) = localstates_[diststate_(GetRandomEngine())];
    }
    log_acceptance_correction.setZero();
  }
};

}  // namespace netket

#endif
