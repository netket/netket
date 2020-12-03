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

#include "density_matrix_row.hpp"

namespace netket {

using VectorType = DensityMatrixRow::VectorType;
using VisibleConstType = DensityMatrixRow::VisibleConstType;

Complex DensityMatrixRow::LogValSingle(VisibleConstType v_col, const any &lt) {
  // return density_matrix_.LogValSingle(v_row_.replicate(v_col.rows(),1),
  // v_col, lt);
  return density_matrix_.LogValSingle(v_row_, v_col, lt);
}

VectorType DensityMatrixRow::DerLogSingle(VisibleConstType v_col,
                                          const any &lt) {
  return density_matrix_.DerLogSingle(v_row_, v_col, lt);
}

void DensityMatrixRow::SetRow(VisibleConstType v) { v_row_ = v; }

VisibleConstType DensityMatrixRow::GetRow() const { return v_row_; }

int DensityMatrixRow::Npar() const { return density_matrix_.Npar(); }

VectorType DensityMatrixRow::GetParameters() {
  return density_matrix_.GetParameters();
}
void DensityMatrixRow::SetParameters(VectorConstRefType pars) {
  return density_matrix_.SetParameters(pars);
}
int DensityMatrixRow::Nvisible() const {
  return density_matrix_.NvisiblePhysical();
}

bool DensityMatrixRow::IsHolomorphic() const noexcept {
  return density_matrix_.IsHolomorphic();
}

void DensityMatrixRow::Save(const std::string &filename) const {
  return density_matrix_.Save(filename);
}

void DensityMatrixRow::Load(const std::string &filename) {
  return density_matrix_.Load(filename);
}

}  // namespace netket