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

#ifndef NETKET_LOCAL_KERNEL_HPP
#define NETKET_LOCAL_KERNEL_HPP

#include <Eigen/Dense>
#include "Machine/abstract_machine.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Metropolis sampling generating local moves in hilbert space
class LocalKernel {
  std::vector<double> local_states_;

  std::uniform_int_distribution<Index> dist_random_sites_;
  std::uniform_int_distribution<Index> dist_random_states_;

 public:
  explicit LocalKernel(const AbstractMachine& psi) { Init(psi); }

  void Init(const AbstractMachine& psi) {
    if (!psi.GetHilbert().IsDiscrete()) {
      throw InvalidInputError(
          "Local Kernel sampler works only for discrete "
          "Hilbert spaces");
    }

    auto nstates = psi.GetHilbert().LocalSize();
    auto nv = psi.GetHilbert().Size();
    local_states_ = psi.GetHilbert().LocalStates();

    // #RandomValues() relies on the fact that locat_states_ are sorted.
    std::sort(local_states_.begin(), local_states_.end());

    dist_random_sites_ = std::uniform_int_distribution<Index>(0, nv - 1);

    // There are `local_states_.size() - 1` possible values (minus one is
    // because we don't want to stay in the same state). Thus first, we generate
    // a random number in `[0, local_states_.size() - 2]`. Next step is to
    // transform the result to avoid the gap.
    dist_random_states_ = std::uniform_int_distribution<Index>(0, nstates - 2);
  }

  void operator()(Eigen::Ref<const RowMatrix<double>> v,
                  Eigen::Ref<RowMatrix<double>> vnew,
                  Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction) {
    vnew = v;

    for (int i = 0; i < v.rows(); i++) {
      // picking a random site to be changed
      Index si = dist_random_sites_(GetRandomEngine());
      Index rs = dist_random_states_(GetRandomEngine());
      vnew(i, si) = local_states_[rs + (local_states_[rs] >= v(i, si))];
    }

    log_acceptance_correction.setZero();
  }
};

}  // namespace netket

#endif
