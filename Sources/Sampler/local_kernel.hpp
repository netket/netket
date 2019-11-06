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
  const Index n_states_;
  const Index nv_;

 public:
  explicit LocalKernel(const AbstractMachine& psi);

  explicit LocalKernel(const AbstractHilbert& psi);

  void operator()(Eigen::Ref<const RowMatrix<double>> v,
                  Eigen::Ref<RowMatrix<double>> vnew,
                  Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction);
};

}  // namespace netket

#endif
