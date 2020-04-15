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

#ifndef NETKET_HAMILTONIAN_KERNEL_HPP
#define NETKET_HAMILTONIAN_KERNEL_HPP

#include <Eigen/Core>
#include "Machine/abstract_machine.hpp"
#include "Operator/abstract_operator.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Generating transitions using the Hamiltonian matrix elements
class HamiltonianKernel {
  AbstractOperator &hamiltonian_;

  // number of visible units
  const Index nv_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

 public:
  HamiltonianKernel(const AbstractMachine &psi, AbstractOperator &hamiltonian);

  HamiltonianKernel(AbstractOperator &ham);

  void operator()(Eigen::Ref<const RowMatrix<double>> v,
                  Eigen::Ref<RowMatrix<double>> vnew,
                  Eigen::Ref<Eigen::ArrayXd> acceptance_correction);
};

}  // namespace netket

#endif
