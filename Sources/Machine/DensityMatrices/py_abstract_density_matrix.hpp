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

#ifndef NETKET_PY_ABSTRACT_DENSITY_MATRIX_HPP
#define NETKET_PY_ABSTRACT_DENSITY_MATRIX_HPP

#include "Machine/DensityMatrices/abstract_density_matrix.hpp"
#include "Machine/py_abstract_machine.hpp"

namespace netket {
template <class AbstractDensityMatrixBase = AbstractDensityMatrix>
class PyAbstractDensityMatrix
    : public PyAbstractMachine<AbstractDensityMatrixBase> {
 public:
  using PyAbstractMachine<
      AbstractDensityMatrixBase>::PyAbstractMachine;  // Inherit construtors
  using PyAbstractMachine = typename PyAbstractMachine<
      AbstractDensityMatrixBase>::PyAbstractMachine;  // Inherit construtors
  using VectorType = typename PyAbstractMachine::VectorType;
  using VectorRefType = typename PyAbstractMachine::VectorRefType;
  using VectorConstRefType = typename PyAbstractMachine::VectorConstRefType;
  using VisibleConstType = typename PyAbstractMachine::VisibleConstType;

  PyAbstractDensityMatrix(std::shared_ptr<const AbstractHilbert> hilbert)
      : PyAbstractMachine{std::move(hilbert)} {};

  int NvisiblePhysical() const override {
    return AbstractDensityMatrixBase::GetHilbertPhysical().Size();
  }

  Complex LogValSingle(VisibleConstType vr, VisibleConstType vc,
                       const any& /*unused*/) override;
  Complex LogValSingle(VisibleConstType vr, const any& /*unused*/) override;
  VectorType DerLogSingle(VisibleConstType vr, VisibleConstType vc,
                          const any& lt) override {
    Eigen::VectorXcd out(PyAbstractMachine::Npar());
    DerLog(vr.transpose(), vc.transpose(),
           Eigen::Map<RowMatrix<Complex>>{out.data(), 1, out.size()}, lt);
    return out;
  };

  ~PyAbstractDensityMatrix() override = default;

  /// Functions which one needs to override from Python
  void LogVal(Eigen::Ref<const RowMatrix<double>> vr,
              Eigen::Ref<const RowMatrix<double>> vc, Eigen::Ref<VectorXcd> out,
              const any& /*unused*/) override;
  void DerLog(Eigen::Ref<const RowMatrix<double>> vr,
              Eigen::Ref<const RowMatrix<double>> vc,
              Eigen::Ref<RowMatrix<Complex>> out, const any& cache) override;
};
}  // namespace netket

#include "py_abstract_density_matrix.ipp"

#endif  // NETKET_PY_ABSTRACT_DENSITY_MATRIX_HPP
