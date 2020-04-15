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
#include "Hilbert/doubled_hilbert.hpp"
#include "Machine/abstract_machine.hpp"
#include "Utils/memory_utils.hpp"

namespace netket {

/**
 * Abstract base class for Density Matrices.
 * Contains the physical hilbert space and the doubled hilbert
 * space where operators are defined.
 */
class AbstractDensityMatrix : public AbstractMachine {
 public:
  using Edge = AbstractGraph::Edge;

  /**
   * Constructs the Abstract base class AbstractDensityMatrix starting from a
   * physical (non-doubled) Hilbert space. This method constructs the doubled
   * hilbert space.
   * @param hilbert : a physical Hilbert space.
   */
  explicit AbstractDensityMatrix(std::shared_ptr<const AbstractHilbert> hilbert)
      : AbstractMachine(std::make_shared<DoubledHilbert>(hilbert)){};

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
   * Member function returning the number of visible units of the rows or
   * columns (equivalent to Nvisible()/2).
   * @return
   */
  virtual int NvisiblePhysical() const = 0;

  // The methods below shadow the declarations in AbstractMachine. We are aware
  // of this, and it's because when we store a density matrix as such, we want
  // to be able to use it's interface with row and column states.
  // The standard AbstractMachine interface can be called by prepending a
  // namespace specifier AbstractMachine:: to a function call.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
  /**
   * Computes logarithm of the density matrix for a batch of visible
   * configurations. Non-allocating version.
   *
   * @param vr : a matrix of size `batch_size x NvisiblePhysical()`. Each row
   * of the matrix is a visible configuration of the rows.
   * @param vc : a matrix of size `batch_size x NvisiblePhysical()`. Each row
   * of the matrix is a visible configuration of the columns.
   * @param out : a pre-allocated vector of length `batch_size` that will store
   * the result of LogVal for every configuration.
   */
  virtual void LogVal(Eigen::Ref<const RowMatrix<double>> vr,
                      Eigen::Ref<const RowMatrix<double>> vc,
                      Eigen::Ref<VectorType> out, const any &);

  /**
   * Computes logarithm of the density matrix for a batch of visible
   * configurations.
   *
   * @param vr : a matrix of size `batch_size x NvisiblePhysical()`. Each row
   * of the matrix is a visible configuration of the rows.
   * @param vc : a matrix of size `batch_size x NvisiblePhysical()`. Each row
   * of the matrix is a visible configuration of the columns.
   * @return a vector of length `batch_size` that contains the result
   * of LogVal for every configuration.
   */
  virtual VectorType LogVal(Eigen::Ref<const RowMatrix<double>> vr,
                            Eigen::Ref<const RowMatrix<double>> vc,
                            const any &);

  /**
   * Computes logarithm of the density matrix for a single input configuration.
   * By default, it internally calls LogVal.
   *
   * @param vr : a Vector of size `NvisiblePhysical()` containing the visible
   * configuration of the row.
   * @param vc : a Vector of size `NvisiblePhysical()` containing the visible
   * configuration of the column
   * @return the logarithm of the density matrix.
   */
  virtual Complex LogValSingle(VisibleConstType vr, VisibleConstType vc,
                               const any &lt = any{}) = 0;

  virtual void DerLog(Eigen::Ref<const RowMatrix<double>> vr,
                      Eigen::Ref<const RowMatrix<double>> vc,
                      Eigen::Ref<RowMatrix<Complex>> out,
                      const any &cache = any{});

  virtual RowMatrix<Complex> DerLog(Eigen::Ref<const RowMatrix<double>> vr,
                                    Eigen::Ref<const RowMatrix<double>> vc,
                                    const any &cache = any{});

  virtual VectorType DerLogSingle(VisibleConstType vr, VisibleConstType vc,
                                  const any &cache = any{}) = 0;

#pragma GCC diagnostic pop
  /**
   * Member function computing the difference between the logarithm of the
   * density-matrix computed at different values of the visible units ((vr,vc),
   * and a set of (vr', vc')).
   * A non-allocating version also exists.
   *
   * @note performance of the default implementation is pretty bad.
   * @note A non-allocating version also exists.
   *
   * @param vr : a constant reference to the current row visible configuration.
   * @param vc : a constant reference to the current column visible
   * configuration.
   * @param tochange_r : a constant reference to a vector containing the indeces
   * of the units to be modified on vr.
   * @param newconf_r : a constant reference to a vector containing the new
   * values of the visible units: here for each vr',
   * newconf_r(i)=vr'(tochange_r(i)), where vr' is the new visible state.
   * @param tochange_c : a constant reference to a vector containing the indeces
   * of the units to be modified on vc.
   * @param newconf_c : a constant reference to a vector containing the new
   * values of the visible units: here for each vr',
   * newconf_c(i)=vc'(tochange_c(i)), where vc' is the new visible state.
   * @return A vector containing, for each (vr',vc'), log(rho(vr', vc')) -
   * log(rho(vr,vc))
   */
   /*
  virtual void LogValDiffRowCol(
      VisibleConstType vr, VisibleConstType vc,
      const std::vector<std::vector<int>> &tochange_r,
      const std::vector<std::vector<double>> &newconf_r,
      const std::vector<std::vector<int>> &tochange_c,
      const std::vector<std::vector<double>> &newconf_c,
      Eigen::Ref<Eigen::VectorXcd> output);

  VectorType LogValDiffRowCol(
      VisibleConstType vr, VisibleConstType vc,
      const std::vector<std::vector<int>> &tochange_r,
      const std::vector<std::vector<double>> &newconf_r,
      const std::vector<std::vector<int>> &tochange_c,
      const std::vector<std::vector<double>> &newconf_c);
  */

  /**
   * Member function computing the difference between the logarithm of the
   * density-matrix computed at different values of the visible units ((vr,vc),
   * and a set of (vr', vc)). This is mainly used to compute \Tr[O\rho], needed
   * for evaluating observables over density matrices.
   *
   * @note performance of the default implementation is pretty bad.
   * @note A non-allocating version also exists.
   *
   * @param vr : a constant reference to the current row visible configuration.
   * @param vc : a constant reference to the current column visible
   * configuration.
   * @param tochange_r : a constant reference to a vector containing the indeces
   * of the units to be modified on vr.
   * @param newconf_r : a constant reference to a vector containing the new
   * values of the visible units: here for each vr',
   * newconf_r(i)=vr'(tochange_r(i)), where vr' is the new visible state.
   *
   * @return A vector containing, for each vr', log(dm(vr', vc)) -
   * log(dm(vr,vc))
   */
  virtual void LogValDiffRow(VisibleConstType vr, VisibleConstType vc,
                             const std::vector<std::vector<int>> &tochange_r,
                             const std::vector<std::vector<double>> &newconf_r,
                             Eigen::Ref<Eigen::VectorXcd> output);

  // Methods below are inherited from AbstractMachine
  int Nvisible() const override { return NvisiblePhysical() * 2; }

  void LogVal(Eigen::Ref<const RowMatrix<double>> v, Eigen::Ref<VectorType> out,
              const any &lt) override;

  Complex LogValSingle(VisibleConstType v, const any &lt) override;

  void DerLog(Eigen::Ref<const RowMatrix<double>> v,
              Eigen::Ref<RowMatrix<Complex>> out, const any &cache) override;

  VectorType DerLogSingle(VisibleConstType v, const any &cache) override;
};

}  // namespace netket

#endif  // NETKET_ABSTRACT_DENSITY_MATRIX_HPP
