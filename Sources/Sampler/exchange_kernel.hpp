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

#ifndef NETKET_EXCHANGE_KERNEL_HPP
#define NETKET_EXCHANGE_KERNEL_HPP

#include <Eigen/Core>
#include <array>
#include "Machine/abstract_machine.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Kernel generating local random exchanges
class ExchangeKernel {
  // number of visible units
  const int nv_;

  // clusters to do updates
  std::vector<std::array<Index, 2>> clusters_;

 public:
  explicit ExchangeKernel(const AbstractMachine &psi, Index dmax = 1);

  void operator()(Eigen::Ref<const RowMatrix<double>> v,
                  Eigen::Ref<RowMatrix<double>> vnew,
                  Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction);

 private:
  void Init(const AbstractGraph &graph, int dmax);
};

}  // namespace netket

#endif
