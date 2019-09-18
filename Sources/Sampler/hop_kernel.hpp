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

#ifndef NETKET_HOP_KERNEL_HPP
#define NETKET_HOP_KERNEL_HPP

#include <Eigen/Core>
#include "Graph/abstract_graph.hpp"
#include "Machine/abstract_machine.hpp"
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

  std::uniform_int_distribution<Index> diststate_;

 public:
  HopKernel(const AbstractMachine &psi, Index dmax = 1);

  void operator()(Eigen::Ref<const RowMatrix<double>> v,
                  Eigen::Ref<RowMatrix<double>> vnew,
                  Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction);

 private:
  void GenerateClusters(const AbstractGraph &graph, Index dmax);
};

}  // namespace netket

#endif
