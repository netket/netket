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

VisibleType DiagonalDensityMatrix::DoubleVisibleConfig(const VisibleConstType v) const {
  VisibleType v2(2 * v.rows());
  v2.head(v.rows()) = v;
  v2.tail(v.rows()) = v;

  return v2;
}

VisibleChangeInfo DiagonalDensityMatrix::DoubleVisibleChangeInfo(const std::vector<int> &tochange,
                                          const std::vector<double> &newconf,
                                          int offset) const {
  std::vector<int> tochange_doubled(tochange.size() * 2);
  std::vector<double> newconf_doubled(newconf.size() * 2);

  // Copy tochange on the first half of tochange_doubled and copy + offset on
  // the other half.
  std::copy(tochange.begin(), tochange.end(), tochange_doubled.begin());
  std::copy(tochange.begin(), tochange.end(),
            tochange_doubled.begin() + tochange.size());
  for (auto tcd = tochange_doubled.begin() + tochange.size();
       tcd != tochange_doubled.end(); ++tcd) {
    *tcd += offset;
  }

  std::copy(newconf.begin(), newconf.end(), newconf_doubled.begin());
  std::copy(newconf.begin(), newconf.end(),
            newconf_doubled.begin() + newconf.size());

  return VisibleChangeInfo(tochange_doubled, newconf_doubled);
};

Complex DiagonalDensityMatrix::LogValSingle(VisibleConstType v, const any &lt)  {
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
  return density_matrix_.Nvisible();
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