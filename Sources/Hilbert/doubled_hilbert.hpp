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

#ifndef NETKET_DOUBLED_HILBERT_HPP
#define NETKET_DOUBLED_HILBERT_HPP

#include <memory>
#include <utility>

#include "Graph/abstract_graph.hpp"
#include "Graph/doubled_graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"

namespace netket {

/**
 * Concrete class for the doubled hilbert space, obtained through Choi's
 * isomorphism, where density operators live.
 */
class DoubledHilbert : public AbstractHilbert {
  // Pointer to the physical (not doubled) hilbert space H.
  std::shared_ptr<const AbstractHilbert> hilbert_physical_;

  // Pointer to the doubled graph, which encodes the correct isomoprhisms
  // along rows and columns.
  std::unique_ptr<const AbstractGraph> graph_doubled_;

  // Number of (doubled) sites
  int size_;

 protected:
  DoubledHilbert(std::shared_ptr<const AbstractHilbert> hilbert,
                 std::unique_ptr<const AbstractGraph> doubled_graph);

 public:
  /**
   * Construct the doubled Hilbert space (H \otimes H) given the standard
   * (non-doubled) Hilbert space H.
   * @param hilbert space H
   */
  explicit DoubledHilbert(const std::shared_ptr<const AbstractHilbert> &hilbert)
      : DoubledHilbert(hilbert, DoubledGraph(hilbert->GetGraph())){};

  // -- DoubledHilbert Interface  -- //
  // Methods below are custom methods for working with density operators.

  int SizePhysical() const;
  /**
   * Member function to access the graph of the physical system
   * @return the graph of the physical system.
   */
  const AbstractGraph &GetGraphPhysical() const noexcept;

  /**
   * Member function to access the physical hilbert space
   * @return hilbert_physical_
   */
  const AbstractHilbert &GetHilbertPhysical() const noexcept;

  /**
   * Member function to access the physical hilbert space as a shared pointer.
   * @return a shared_ptr to hilbert_physical_
   */
  std::shared_ptr<const AbstractHilbert> GetHilbertPhysicalShared() const;

  /**
   * Member function to call RandomVals only on the part of the state describing
   * the row.
   * @param state : a state of the doubled hilbert space
   * @param rgen the random number generator to be used
   */
  void RandomValsRows(Eigen::Ref<Eigen::VectorXd> state,
                      netket::default_random_engine &rgen) const;

  /**
   * Member function to call RandomVals only on the part of the state describing
   * the column.
   * @param state : a state of the doubled hilbert space
   * @param rgen the random number generator to be used
   */
  void RandomValsCols(Eigen::Ref<Eigen::VectorXd> state,
                      netket::default_random_engine &rgen) const;

  /**
   * Member function to call RandomVals on the state of the physical system.
   * Equivalent to this.GetHilbertPhysical.RandomVals(state, rgen).
   * @param state : a state of the doubled hilbert space
   * @param rgen the random number generator to be used
   */
  void RandomValsPhysical(Eigen::Ref<Eigen::VectorXd> state,
                          default_random_engine &rgen) const;

  /**
   * Member function updating a visible configuration using the information on
   * where the local changes have been done on the part encoding either the rows
   * or columns of the state.
   * Equivalent to this.GetHilbertPhysical.UpdateConf(v, tochange, newconf).
   * @param v is the vector visible units to be modified.
   * @param tochange contains a list of which quantum numbers are to be
   * modified.
   * @param newconf contains the value that those quantum numbers should take
   */
  void UpdateConfPhysical(Eigen::Ref<Eigen::VectorXd> v,
                          nonstd::span<const int> tochange,
                          nonstd::span<const double> newconf) const;

  // -- AbstractHilbert interface -- //
  // Methods below are inherited from AbstractHilbert.

  bool IsDiscrete() const override;
  int LocalSize() const override;
  int Size() const override;
  std::vector<double> LocalStates() const override;

  /**
   * Member function to access the doubled graph
   * @return the doubled graph graph_doubled_
   */
  const AbstractGraph &GetGraph() const noexcept override;

  void RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                  netket::default_random_engine &rgen) const override;

  void UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                  nonstd::span<const int> tochange,
                  nonstd::span<const double> newconf) const override;
};
}  // namespace netket

#endif  // NETKET_DOUBLED_HILBERT_HPP
