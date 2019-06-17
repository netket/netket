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

#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <fstream>
#include <utility>
#include <vector>
#include "Machine/abstract_machine.hpp"
#include "Utils/lookup.hpp"
#include "Utils/random_utils.hpp"
#include "density_matrix_machine.hpp"

namespace netket {
class DiagonalDensityMatrix : public AbstractMachine {
  using VisibleChangeInfo = std::pair<std::vector<int>, std::vector<double>>;

  AbstractDensityMatrix &density_matrix_;

 public:
  explicit DiagonalDensityMatrix(AbstractDensityMatrix &dm);

  const AbstractDensityMatrix &GetFullDensityMatrix() const noexcept {
    return density_matrix_;
  }

 public:
  Complex LogVal(VisibleConstType v) override;
  Complex LogVal(VisibleConstType v, const LookupType &lt) override;
  void InitLookup(VisibleConstType v, LookupType &lt) override;

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override ;

  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override ;

  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override ;

  VectorType DerLog(VisibleConstType v) override ;
  VectorType DerLog(VisibleConstType v, const LookupType &lt) override ;

  VectorType DerLogChanged(VisibleConstType v, const std::vector<int> &tochange,
                           const std::vector<double> &newconf) override ;

  int Npar() const override { return density_matrix_.Npar(); }

  VectorType GetParameters() override {
    return density_matrix_.GetParameters();
  }

  void SetParameters(VectorConstRefType pars) override {
    return density_matrix_.SetParameters(pars);
  }

  void InitRandomPars(int seed, double sigma) override {
    return density_matrix_.InitRandomPars(seed, sigma);
  }

  int Nvisible() const override { return density_matrix_.Nvisible(); }

  bool IsHolomorphic() const noexcept override {
    return density_matrix_.IsHolomorphic();
  }

  void Save(const std::string &filename) const override {
    return density_matrix_.Save(filename);
  }

  void Load(const std::string &filename) override {
    return density_matrix_.Load(filename);
  }
};
}  // namespace netket

#endif  // NETKET_DIAGONALDENSITYMATRIX_HPP
