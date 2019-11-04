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

#include "local_kernel.hpp"

namespace netket {

LocalKernel::LocalKernel(const AbstractMachine& psi)
    : local_states_(psi.GetHilbert().LocalStates()),
      n_states_(psi.GetHilbert().LocalSize()),
      nv_(psi.GetHilbert().Size()) {
  NETKET_CHECK(psi.GetHilbert().IsDiscrete(), InvalidInputError,
               "Local Kernel sampler works only for discrete "
               "Hilbert spaces");

  // operator() relies on the fact that locat_states_ are sorted.
  std::sort(local_states_.begin(), local_states_.end());
}

LocalKernel::LocalKernel(std::vector<double> local_states, Index n_visible)
    : local_states_(local_states),
      n_states_(local_states.size()),
      nv_(n_visible) {
  // operator() relies on the fact that locat_states_ are sorted.
  std::sort(local_states_.begin(), local_states_.end());
}

void LocalKernel::operator()(
    Eigen::Ref<const RowMatrix<double>> v, Eigen::Ref<RowMatrix<double>> vnew,
    Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction) {
  vnew = v;

  for (int i = 0; i < v.rows(); i++) {
    // picking a random site to be changed
    Index si =
        std::uniform_int_distribution<Index>(0, nv_ - 1)(GetRandomEngine());

    // There are `local_states_.size() - 1` possible values (minus one is
    // because we don't want to stay in the same state). Thus first, we
    // generate a random number in `[0, local_states_.size() - 2]`. Next step
    // is to transform the result to avoid the gap.
    Index rs = std::uniform_int_distribution<Index>(
        0, n_states_ - 2)(GetRandomEngine());

    vnew(i, si) = local_states_[rs + (local_states_[rs] >= v(i, si))];
  }

  log_acceptance_correction.setZero();
}

}  // namespace netket
