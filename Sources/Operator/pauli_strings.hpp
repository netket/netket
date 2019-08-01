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

#ifndef NETKET_PAULISTRINGS_HPP
#define NETKET_PAULISTRINGS_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <vector>
#include "Graph/edgeless.hpp"
#include "abstract_operator.hpp"

namespace netket {

// Pauli operator : i.e. sum of products of Pauli matrices

class PauliStrings : public AbstractOperator {
  const int nqubits_;
  const int noperators_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<Complex>> weights_;

  std::vector<std::vector<std::vector<int>>> zcheck_;

  const Complex I_;

  double cutoff_;

 public:
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

  PauliStrings(const std::vector<std::string> &ops,
               const std::vector<Complex> &opweights, double cutoff);

  Edgeless GraphFromOps(const std::vector<std::string> &ops);

  Index CheckOps(const std::vector<std::string> &ops);

  void FindConn(VectorConstRefType v, std::vector<Complex> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override;
};

}  // namespace netket

#endif
