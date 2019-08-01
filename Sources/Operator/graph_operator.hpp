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

#ifndef NETKET_GRAPH_OPERATOR_HPP
#define NETKET_GRAPH_OPERATOR_HPP

#include <Eigen/Dense>
#include <array>
#include <unordered_map>
#include <vector>
#include "Graph/abstract_graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"
#include "abstract_operator.hpp"
#include "local_operator.hpp"

namespace netket {

// Graph Hamiltonian on an arbitrary graph
class GraphOperator : public AbstractOperator {
  // Arbitrary graph
  const AbstractGraph &graph_;

  LocalOperator operator_;

  const int nvertices_;

 public:
  using SiteType = std::vector<int>;
  using OMatType = LocalOperator::MatType;
  using OVecType = std::vector<OMatType>;
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

  GraphOperator(std::shared_ptr<const AbstractHilbert> hilbert,
                OVecType siteops, OVecType bondops,
                std::vector<int> bondops_colors);

  // Constructor to be used when overloading operators
  GraphOperator(std::shared_ptr<const AbstractHilbert> hilbert,
                const LocalOperator &lop);

  void FindConn(VectorConstRefType v, std::vector<Complex> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override;

  void ForEachConn(VectorConstRefType v, ConnCallback callback) const override;

  friend GraphOperator operator+(const GraphOperator &lhs,
                                 const GraphOperator &rhs);
};  // namespace netket
}  // namespace netket
#endif
