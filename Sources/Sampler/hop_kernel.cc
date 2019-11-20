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

#include "hop_kernel.hpp"

namespace netket {

HopKernel::HopKernel(const AbstractMachine &psi, Index dmax)
    : nv_(psi.GetHilbert().Size()),
      nstates_(psi.GetHilbert().LocalSize()),
      localstates_(psi.GetHilbert().LocalStates()),
      diststate_(0, nstates_ - 1) {
  NETKET_CHECK(dmax >= 1, InvalidInputError,
               "d_max should be at least equal to 1");
  GenerateClusters(psi.GetHilbert().GetGraph(), dmax);
}

HopKernel::HopKernel(const AbstractHilbert &hilb, Index dmax)
    : nv_(hilb.Size()),
      nstates_(hilb.LocalSize()),
      localstates_(hilb.LocalStates()),
      diststate_(0, nstates_ - 1) {
  NETKET_CHECK(dmax >= 1, InvalidInputError,
               "d_max should be at least equal to 1");
  GenerateClusters(hilb.GetGraph(), dmax);
}

void HopKernel::GenerateClusters(const AbstractGraph &graph, Index dmax) {
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
}

void HopKernel::operator()(
    Eigen::Ref<const RowMatrix<double>> v, Eigen::Ref<RowMatrix<double>> vnew,
    Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction) {
  vnew = v;

  for (int i = 0; i < v.rows(); i++) {
    Index rcl = std::uniform_int_distribution<Index>(
        0, clusters_.size() - 1)(GetRandomEngine());
    assert(rcl < Index(clusters_.size()));
    Index si = clusters_[rcl][0];
    Index sj = clusters_[rcl][1];

    assert(si < nv_ && sj < nv_);

    vnew(i, si) = localstates_[diststate_(GetRandomEngine())];
    vnew(i, sj) = localstates_[diststate_(GetRandomEngine())];
  }
  log_acceptance_correction.setZero();
}

}  // namespace netket
