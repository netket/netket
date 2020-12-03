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

#ifndef NETKET_DENSITYMATRIX_ROW_HPP
#define NETKET_DENSITYMATRIX_ROW_HPP

#include "Machine/abstract_machine.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_density_matrix.hpp"

#include <Eigen/Core>
#include <functional>
#include <limits>
#include <vector>

namespace netket {
class DensityMatrixRow : public AbstractMachine {

  AbstractDensityMatrix &density_matrix_;

  Eigen::VectorXd v_row_;

 public:
  explicit DensityMatrixRow(AbstractDensityMatrix &dm, VisibleConstType v_row)
      : AbstractMachine(dm.GetHilbertPhysicalShared()), density_matrix_(dm) {
        v_row_.resize(dm.GetHilbertPhysical().Size());
        SetRow(v_row);
      } ;

  explicit DensityMatrixRow(AbstractDensityMatrix &dm)
      : AbstractMachine(dm.GetHilbertPhysicalShared()), density_matrix_(dm) {
        v_row_.resize(dm.GetHilbertPhysical().Size());
      } ;

  const AbstractDensityMatrix &GetFullDensityMatrix() const noexcept {
    return density_matrix_;
  }

  Complex LogValSingle(VisibleConstType v_col, const any &lt) override;

  VectorType DerLogSingle(VisibleConstType v, const any &lt) override;

  void SetRow(VisibleConstType v);

  VisibleConstType GetRow() const;

  int Npar() const override;

  VectorType GetParameters() override;

  void SetParameters(VectorConstRefType pars) override;

  int Nvisible() const override ;

  bool IsHolomorphic() const noexcept override ;

  void Save(const std::string &filename) const override ;

  void Load(const std::string &filename) override ;
};
}  // namespace netket

#endif  // NETKET_DENSITYMATRIX_ROW_HPP
