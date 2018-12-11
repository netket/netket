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

#ifndef NETKET_GRAPH_HAMILTONIAN_CC
#define NETKET_GRAPH_HAMILTONIAN_CC

#include <Eigen/Dense>
#include <array>
#include <unordered_map>
#include <vector>
#include "Graph/graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"
#include "Utils/json_helper.hpp"
#include "abstract_operator.hpp"
#include "local_operator.hpp"

namespace netket {

// Graph Hamiltonian on an arbitrary graph
class GraphOperator : public AbstractOperator {
  const AbstractHilbert &hilbert_;

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

  explicit GraphOperator(const AbstractHilbert &hilbert,
                         OVecType siteops = OVecType(),
                         OVecType bondops = OVecType(),
                         std::vector<int> bondops_colors = std::vector<int>())
      : hilbert_(hilbert),
        graph_(hilbert.GetGraph()),
        operator_(LocalOperator(hilbert)),
        nvertices_(hilbert.Size()) {
    // Create the local operator as the sum of all site and bond operators
    // Ensure that at least one of SiteOps and BondOps was initialized
    if (!siteops.size() && !bondops.size()) {
      throw InvalidInputError("Must input at least SiteOps or BondOps");
    }

    std::vector<int> op_color;
    if (bondops_colors.size() == 0) {
      op_color = std::vector<int>(bondops.size(), 0);
    } else {
      op_color = bondops_colors;
    }

    // Site operators
    if (siteops.size() > 0) {
      for (int i = 0; i < nvertices_; i++) {
        for (std::size_t j = 0; j < siteops.size(); j++) {
          operator_ = operator_ +
                      LocalOperator(hilbert_, siteops[j], std::vector<int>{i});
        }
      }
    }

    // Bond operators
    if (bondops.size() != op_color.size()) {
      throw InvalidInputError(
          "The bond Hamiltonian definition is inconsistent."
          "The sizes of BondOps and BondOpColors do not match.");
    }

    if (bondops.size() > 0) {
      // Use EdgeColors to populate operators
      for (auto const &kv : graph_.EdgeColors()) {
        for (std::size_t c = 0; c < op_color.size(); c++) {
          if (op_color[c] == kv.second && kv.first[0] < kv.first[1]) {
            std::vector<int> edge = {kv.first[0], kv.first[1]};
            operator_ = operator_ + LocalOperator(hilbert_, bondops[c], edge);
          }
        }
      }
    }

    // InfoMessage() << "Size of operators_ " << operators_.size() << std::endl;
  }

  // Constructor to be used when overloading operators
  explicit GraphOperator(const AbstractHilbert &hilbert, LocalOperator lop)
      : hilbert_(hilbert),
        graph_(hilbert.GetGraph()),
        operator_(lop),
        nvertices_(hilbert.Size()) {}

  friend GraphOperator operator+(const GraphOperator &lhs,
                                 const GraphOperator &rhs) {
    assert(rhs.graph_.Size() == lhs.graph_.Size());

    auto lop = lhs.operator_;
    auto rop = rhs.operator_;

    return GraphOperator(lhs.GetHilbert(), lop + rop);
  }

  void FindConn(VectorConstRefType v, std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    connectors.clear();
    newconfs.clear();
    mel.resize(0);

    operator_.FindConn(v, mel, connectors, newconfs);
  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }
};  // namespace netket
}  // namespace netket
#endif
