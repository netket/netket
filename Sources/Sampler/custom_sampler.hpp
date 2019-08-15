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

// authors: Hugo Théveniaut and Fabien Alet

#ifndef NETKET_CUSTOMSAMPLER_HPP
#define NETKET_CUSTOMSAMPLER_HPP

#include <Eigen/Core>
#include <set>
#include "Operator/local_operator.hpp"

#include "Machine/abstract_machine.hpp"
#include "Utils/exceptions.hpp"
#include "Utils/messages.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Metropolis sampling using custom moves provided by user
class CustomSampler : public AbstractSampler {
  LocalOperator move_operators_;
  std::vector<double> operatorsweights_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  int nstates_;
  std::vector<double> localstates_;

  int sweep_size_;

 public:
  CustomLocalKernel(const AbstractMachine &psi,
                    const LocalOperator &move_operators,
                    const std::vector<double> &move_weights = {});

  void operator()(Eigen::Ref<const RowMatrix<double>> v,
                  Eigen::Ref<RowMatrix<double>> vnew,

                  Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction);

 private:
  void Init(const AbstractMachine &psi,
            const std::vector<double> &move_weights);
  void CheckMoveOperators(const LocalOperator &move_operators);
};
}  // namespace netket

#endif
