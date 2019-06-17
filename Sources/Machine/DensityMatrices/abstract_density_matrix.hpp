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

#include "Graph/custom_graph.hpp"
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

 public:
  using ChangeInfo = std::tuple<std::vector<int>, std::vector<double>>;
  using RowColChangeInfo = std::tuple<ChangeInfo, ChangeInfo>;

 protected:
  AbstractDensityMatrix(std::shared_ptr<const AbstractHilbert> physical_hilbert,
                        std::unique_ptr<const AbstractGraph> doubled_graph);

 public:
  explicit AbstractDensityMatrix(
      std::shared_ptr<const AbstractHilbert> hilbert);

  /**
   * Member function returning the physical hilbert space over which
   * this density matrix acts as a shared pointer.
   * @return The physical hilbert space
   */
  std::shared_ptr<const AbstractHilbert> GetHilbertPhysicalShared() const;

  /**
   * Member function returning a reference to the physical hilbert space
   * on which this density matrix acts.
   * @return The physical hilbert space
   */
  const AbstractHilbert &GetHilbertPhysical() const noexcept;

  /**
  Member function initializing the look-up tables.
  If needed, a Machine can make use of look-up tables
  to speed up some critical functions. For example,
  to speed up the calculation of density-matrix ratios.
  The state of a look-up table depends on the visible units.
  This function should initialize the look-up tables
  making sure that memory in the table is also being allocated.
  @param v_r a constant reference to the visible configuration of the row.
  @param v_c a constant reference to the visible configuration of the column.
  @param lt a reference to the look-up table to be initialized.
  */
  virtual void InitLookup(VisibleConstType v_r, VisibleConstType v_c,
                          LookupType &lt) = 0;
  /**
  Member function updating the look-up tables.
  If needed, a Machine can make use of look-up tables
  to speed up some critical functions. For example,
  to speed up the calculation of density matrix ratios.
  The state of a look-up table depends on the visible units.
  This function should update the look-up tables
  when the state of visible units is changed according
  to the information stored in toflip and newconf
  @param v_r a constant reference to the visible configuration of the row.
  @param v_c a constant reference to the visible configuration of the column.
  @param tochange_r a constant reference to a vector containing the indices of
  the units to be modified on the row configuration.
  @param tochange_c a constant reference to a vector containing the indices of
  the units to be modified on the column configuration.
  @param newconf_r a constant reference to a vector containing the new values of
  the row visible units: here newconf_r(i)=v_r'(tochange_r(i)), where v_r' is
  the new row visible state.
  @param newconf_c a constant reference to a vector containing the new values of
  the column visible units: here newconf_c(i)=v_c'(tochange_c(i)), where v_c' is
  the new column visible state.
  @param lt a reference to the look-up table to be updated.
  */
  virtual void UpdateLookup(VisibleConstType v_r, VisibleConstType v_c,
                            const std::vector<int> &tochange_r,
                            const std::vector<int> &tochange_c,
                            const std::vector<double> &newconf_r,
                            const std::vector<double> &newconf_c,
                            LookupType &lt) = 0;

  /**
  Member function computing the derivative of the logarithm of the
  density-matrix function for a given visible vector.
  @param v_r a constant reference to the visible configuration of the row.
  @param v_c a constant reference to the visible configuration of the column.
  @return Derivatives of the logarithm of the density matrix with respect to the
  set of parameters.
  */
  virtual VectorType DerLog(VisibleConstType v_r, VisibleConstType v_c) = 0;

  /**
  Member function computing the derivative of the logarithm of the
  density-matrix for a given visible vector. This specialized version, if
  implemented, should make use of the Lookup table to speed up the calculation.
  @param v_r a constant reference to the visible configuration of the row.
  @param v_c a constant reference to the visible configuration of the column.
  @return Derivatives of the logarithm of the density matrix with respect to the
  set of parameters.
  */
  virtual VectorType DerLog(VisibleConstType v_r, VisibleConstType v_c,
                            const LookupType &lt) = 0;
  /**
  Member function computing the logarithm of the density matrix for the given
  row and column visible vectors. Given the current set of parameters, this
  function should compute the value of the logarithm of the density matrix from
  scratch.
  @param v_r a constant reference to the visible configuration of the row.
  @param v_c a constant reference to the visible configuration of the column.
  @return Logarithm of the density matrix function.
  */
  virtual Complex LogVal(VisibleConstType v_r, VisibleConstType v_c) = 0;

  /**
  Member function computing the logarithm of the density matrix for the given
  row and column visible vectors. Given the current set of parameters, this
  function should compute the value of the logarithm of the density matrix using
  the information provided in the look-up table, to speed up the computation.
  @param v_r a constant reference to the visible configuration of the row.
  @param v_c a constant reference to the visible configuration of the column.
  @param lt a constant eference to the look-up table.
  @return Logarithm of the density matrix function.
  */
  virtual Complex LogVal(VisibleConstType v_r, VisibleConstType v_c,
                         const LookupType &lt) = 0;

  /**
  Member function computing the difference between the logarithm of the
  density matrix computed at different values of the visible units ((v_r, v_c),
  and a set of (v_r',v_c')).
  @param v_r a constant reference to the visible configuration of the row.
  @param v_c a constant reference to the visible configuration of the column.
  @param tochange_r a constant reference to a vector containing the indices of
  the units to be modified on the row configuration.
  @param tochange_c a constant reference to a vector containing the indices of
  the units to be modified on the column configuration.
  @param newconf_r a constant reference to a vector containing the new values of
  the row visible units: here newconf_r(i)=v_r'(tochange_r(i)), where v_r' is
  the new row visible state.
  @param newconf_c a constant reference to a vector containing the new values of
  the column visible units: here newconf_c(i)=v_c'(tochange_c(i)), where v_c' is
  the new column visible state.
  @return A vector containing, for each ( v_r',v_c'),log(Rho(v_r', v_c')) -
  log(Rho(v_r', v_c'))
  */
  virtual VectorType LogValDiff(
      VisibleConstType v_r, VisibleConstType v_c,
      const std::vector<std::vector<int>> &tochange_r,
      const std::vector<std::vector<int>> &tochange_c,
      const std::vector<std::vector<double>> &newconf_r,
      const std::vector<std::vector<double>> &newconf_c) = 0;

  /**
  Member function computing the difference between the logarithm of the
  wave-function computed at different values of the visible units ((v_r, v_c),
  and a single (v_r',v_c')). This version uses the look-up tables to speed-up
  the calculation.
  @param v_r a constant reference to the visible configuration of the row.
  @param v_c a constant reference to the visible configuration of the column.
  @param tochange_r a constant reference to a vector containing the indices of
  the units to be modified on the row configuration.
  @param tochange_c a constant reference to a vector containing the indices of
  the units to be modified on the column configuration.
  @param newconf_r a constant reference to a vector containing the new values of
  the row visible units: here newconf_r(i)=v_r'(tochange_r(i)), where v_r' is
  the new row visible state.
  @param newconf_c a constant reference to a vector containing the new values of
  the column visible units: here newconf_c(i)=v_c'(tochange_c(i)), where v_c' is
  the new column visible state.
  @param lt a constant reference to the look-up table.
  @return The value of log(Rho(v_r', v_c')) - log(Rho(v_r', v_c'))
*/
  virtual Complex LogValDiff(VisibleConstType v_r, VisibleConstType v_c,
                             const std::vector<int> &tochange_r,
                             const std::vector<int> &tochange_c,
                             const std::vector<double> &newconf_r,
                             const std::vector<double> &newconf_c,
                             const LookupType &lt) = 0;
  /**
  Member function computing O_k(v'), the derivative of
  the logarithm of the wave function at an update visible state v', given the
  current value at v. Specialized versions use the look-up tables to speed-up
  the calculation, otherwise it is computed from scratch.
  @param v_r a constant reference to the visible configuration of the row.
  @param v_c a constant reference to the visible configuration of the column.
  @param tochange_r a constant reference to a vector containing the indices of
  the units to be modified on the row configuration.
  @param tochange_c a constant reference to a vector containing the indices of
  the units to be modified on the column configuration.
  @param newconf_r a constant reference to a vector containing the new values of
  the row visible units: here newconf_r(i)=v_r'(tochange_r(i)), where v_r' is
  the new row visible state.
  @param newconf_c a constant reference to a vector containing the new values of
  the column visible units: here newconf_c(i)=v_c'(tochange_c(i)), where v_c' is
  the new column visible state.
  @param lt a constant reference to the look-up table.
  @return The value of DerLog(v')
*/
  virtual VectorType DerLogChanged(VisibleConstType v_r, VisibleConstType v_c,
                                   const std::vector<int> &tochange_r,
                                   const std::vector<int> &tochange_c,
                                   const std::vector<double> &newconf,
                                   const std::vector<double> &newconf_c);

  // Implementations for the AbstractMachine interface.
  void InitLookup(VisibleConstType v, LookupType &lt) override;
  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override;
  VectorType DerLog(VisibleConstType v) override;
  VectorType DerLog(VisibleConstType v, const LookupType &lt) override;
  Complex LogVal(VisibleConstType v) override;
  Complex LogVal(VisibleConstType v, const LookupType &lt) override;
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override;
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override;
  using AbstractMachine::DerLogChanged;

 protected:
  /**
   * Returns the disjoint union G of a graph g with itself. The resulting graph
   * has twice the number of vertices and edges of g.
   * The automorpisms of G are the automorphisms of g applied identically
   * to both it's subgraphs.
   */
  static std::unique_ptr<CustomGraph> DoubledGraph(const AbstractGraph &graph);
  RowColChangeInfo SplitRowColsChange(const std::vector<int> &tochange,
                                      const std::vector<double> &newconf) const;
};
}  // namespace netket

#endif  // NETKET_ABSTRACT_DENSITY_MATRIX_HPP
