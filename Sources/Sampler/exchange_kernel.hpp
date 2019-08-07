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

#ifndef NETKET_EXCHANGE_KERNEL_HPP
#define NETKET_EXCHANGE_KERNEL_HPP

#include <Eigen/Core>
#include "Sampler/abstract_sampler.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Kernel generating local random exchanges
class ExchangeKernel {
  // number of visible units
  const int nv_;

  // clusters to do updates
  std::vector<std::vector<Index>> clusters_;

  std::uniform_int_distribution<Index> distcl_;

 public:
  explicit ExchangeKernel(const AbstractMachine &psi, Index dmax = 1)
      : nv_(psi.GetHilbert().Size()) {
    Init(psi.GetHilbert().GetGraph(), dmax);
  }

  template <class G>
  void Init(const G &graph, int dmax) {
    GenerateClusters(graph, dmax);

    InfoMessage() << "Exchange Kernel is ready " << std::endl;
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

    distcl_ = std::uniform_int_distribution<Index>(0, clusters_.size() - 1);
  }

  void operator()(Eigen::Ref<const RowMatrix<double>> v,
                  Eigen::Ref<RowMatrix<double>> vnew,
                  Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction,
                  default_random_engine &random_engine) {
    vnew = v;

    for (int i = 0; i < v.rows(); i++) {
      Index rcl = distcl_(random_engine);

      Index si = clusters_[rcl][0];
      Index sj = clusters_[rcl][1];

      assert(si < nv_ && sj < nv_);

      std::swap(vnew(i, si), vnew(i, sj));
    }

    log_acceptance_correction.setZero();
  }
};

}  // namespace netket

#endif
