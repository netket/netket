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

#include "exchange_kernel.hpp"

namespace netket {

ExchangeKernel::ExchangeKernel(const AbstractMachine &psi, Index dmax)
    : nv_(psi.GetHilbert().Size()) {
  Init(psi.GetHilbert().GetGraph(), dmax);
}

ExchangeKernel::ExchangeKernel(const AbstractHilbert &hilb, Index dmax)
    : nv_(hilb.Size()) {
  Init(hilb.GetGraph(), dmax);
}

void ExchangeKernel::Init(const AbstractGraph &graph, int dmax) {
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

void ExchangeKernel::operator()(
    Eigen::Ref<const RowMatrix<double>> v, Eigen::Ref<RowMatrix<double>> vnew,
    Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction) {
  vnew = v;

  for (int i = 0; i < v.rows(); i++) {
    Index rcl = std::uniform_int_distribution<Index>(
        0, clusters_.size() - 1)(GetRandomEngine());

    Index si = clusters_[rcl][0];
    Index sj = clusters_[rcl][1];

    assert(si < nv_ && sj < nv_);

    std::swap(vnew(i, si), vnew(i, sj));
  }

  log_acceptance_correction.setZero();
}

}  // namespace netket
