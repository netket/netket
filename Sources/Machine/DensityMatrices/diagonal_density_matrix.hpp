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

#ifndef NETKET_DIAGONALDENSITYMATRIX_HPP
#define NETKET_DIAGONALDENSITYMATRIX_HPP

#include "Machine/abstract_machine.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_density_matrix.hpp"

namespace netket {
class DiagonalDensityMatrix : public AbstractMachine {

  AbstractDensityMatrix &density_matrix_;

 public:
  explicit DiagonalDensityMatrix(AbstractDensityMatrix &dm)
      : AbstractMachine(dm.GetHilbertPhysicalShared()), density_matrix_(dm){};

  const AbstractDensityMatrix &GetFullDensityMatrix() const noexcept {
    return density_matrix_;
  }

  using VisibleChangeInfo = std::pair<std::vector<int>, std::vector<double>>;

 private:
  /**
   * Doubles a visible configuration v so that it represents an element on the
   * diagonal of a density matrix.
   * @param v is a visible configuration
   * @return the vertical concatentation of [v, v]
   */
  VisibleType DoubleVisibleConfig(const VisibleConstType v) const;

  VisibleChangeInfo DoubleVisibleChangeInfo(const std::vector<int> &tochange,
                                            const std::vector<double> &newconf,
                                            int offset) const ;

 public:
  Complex LogValSingle(VisibleConstType vr, const any &lt) override;

  VectorType DerLogSingle(VisibleConstType v, const any &lt) override;

  int Npar() const override;

  VectorType GetParameters() override;

  void SetParameters(VectorConstRefType pars) override;

  int Nvisible() const override ;

  bool IsHolomorphic() const noexcept override ;

  void Save(const std::string &filename) const override ;

  void Load(const std::string &filename) override ;
};
}  // namespace netket

#endif  // NETKET_DIAGONALDENSITYMATRIX_HPP
