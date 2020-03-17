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

#include "diagonal_density_matrix.hpp"

namespace netket {

using VectorType = DiagonalDensityMatrix::VectorType;
using VisibleConstType = DiagonalDensityMatrix::VisibleConstType;
using VisibleType = DiagonalDensityMatrix::VisibleType;
using VisibleChangeInfo = DiagonalDensityMatrix::VisibleChangeInfo;

Complex DiagonalDensityMatrix::LogValSingle(VisibleConstType v, const any &lt) {
  return density_matrix_.LogValSingle(v, v, lt);
}

VectorType DiagonalDensityMatrix::DerLogSingle(VisibleConstType v,
                                               const any &lt) {
  return density_matrix_.DerLogSingle(v, v, lt);
}

int DiagonalDensityMatrix::Npar() const { return density_matrix_.Npar(); }

VectorType DiagonalDensityMatrix::GetParameters() {
  return density_matrix_.GetParameters();
}
void DiagonalDensityMatrix::SetParameters(VectorConstRefType pars) {
  return density_matrix_.SetParameters(pars);
}
int DiagonalDensityMatrix::Nvisible() const {
  return density_matrix_.NvisiblePhysical();
}

bool DiagonalDensityMatrix::IsHolomorphic() const noexcept {
  return density_matrix_.IsHolomorphic();
}

void DiagonalDensityMatrix::Save(const std::string &filename) const {
  return density_matrix_.Save(filename);
}

void DiagonalDensityMatrix::Load(const std::string &filename) {
  return density_matrix_.Load(filename);
}

}  // namespace netket