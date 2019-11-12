// Copyright 2018 The Simons Foundation, Inc. - All
// Rights Reserved.
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

#ifndef NETKET_ABSTRACT_DENSITY_MATRIX_HPP
#define NETKET_ABSTRACT_DENSITY_MATRIX_HPP

#include "Graph/doubled_graph.hpp"
#include "Hilbert/custom_hilbert.hpp"
#include "Machine/abstract_machine.hpp"
#include "Utils/memory_utils.hpp"

namespace netket {

/* Abstract base class for Density Matrices.
 * Contains the physical hilbert space and the doubled hilbert
 * space where operators are defined.
 */
class AbstractDensityMatrix : public AbstractMachine {
  // The physical hilbert space over which this operator acts

  std::shared_ptr<const AbstractHilbert> hilbert_physical_;
  std::unique_ptr<const AbstractGraph> graph_doubled_;

  using Edge = AbstractGraph::Edge;

 protected:
  AbstractDensityMatrix(std::shared_ptr<const AbstractHilbert> physical_hilbert,
                        std::unique_ptr<const AbstractGraph> doubled_graph)
      : AbstractMachine(std::make_shared<CustomHilbert>(
            *doubled_graph, physical_hilbert->LocalStates())),
        hilbert_physical_(physical_hilbert),
        graph_doubled_(std::move(doubled_graph)) {}

 public:
  explicit AbstractDensityMatrix(std::shared_ptr<const AbstractHilbert> hilbert)
      : AbstractDensityMatrix(hilbert, DoubledGraph(hilbert->GetGraph())){};

  /**
   * Member function returning the physical hilbert space over which
   * this density matrix acts as a shared pointer.
   * @return The physical hilbert space
   */
  std::shared_ptr<const AbstractHilbert> GetHilbertPhysicalShared() const {
    return hilbert_physical_;
  }

  /* Member function returning a reference to the physical hilbert space
   * on which this density matrix acts.
   * @return The physical hilbert space
   */
  const AbstractHilbert &GetHilbertPhysical() const noexcept {
    return *hilbert_physical_;
  }

  // Batched version of LogVal
  void LogVal(Eigen::Ref<const RowMatrix<double>> v, Eigen::Ref<VectorType> out,
              const any &) override {
    CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()},
               {std::ignore, 2 * Nvisible()});
    CheckShape(__FUNCTION__, "out", out.size(), v.rows());
    for (auto i = Index{0}; i < v.rows(); ++i) {
      out(i) = LogValSingle(v.row(i));
    }
  }

  // Batched version of DerLog
  void DerLog(Eigen::Ref<const RowMatrix<double>> v,
              Eigen::Ref<RowMatrix<Complex>> out,
              const any & /*cache*/) override {
    CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()},
               {std::ignore, 2 * Nvisible()});
    CheckShape(__FUNCTION__, "out", {out.rows(), out.cols()},
               {v.rows(), Npar()});
    for (auto i = Index{0}; i < v.rows(); ++i) {
      out.row(i) = DerLogSingle(v.row(i));
    }
  }
};
}  // namespace netket

#endif  // NETKET_ABSTRACT_DENSITY_MATRIX_HPP
