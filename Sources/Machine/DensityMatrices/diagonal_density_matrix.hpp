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
  explicit DiagonalDensityMatrix(AbstractDensityMatrix &dm)
      : AbstractMachine(dm.GetHilbertPhysicalShared()), density_matrix_(dm){};

  const AbstractDensityMatrix &GetFullDensityMatrix() const noexcept {
    return density_matrix_;
  }

 private:
  /**
   * Doubles a visible configuration v so that it represents an element on the
   * diagonal of a density matrix.
   * @param v is a visible configuration
   * @return the vertical concatentation of [v, v]
   */
  VisibleType DoubleVisibleConfig(const VisibleConstType v) const {
    VisibleType v2(2 * v.rows());
    v2.head(v.rows()) = v;
    v2.tail(v.rows()) = v;

    return v2;
  }

  VisibleChangeInfo DoubleVisibleChangeInfo(const std::vector<int> &tochange,
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

 public:
  Complex LogVal(VisibleConstType v) override {
    return density_matrix_.LogVal(DoubleVisibleConfig(v));
  }

  Complex LogVal(VisibleConstType v, const LookupType &lt) override {
    return density_matrix_.LogVal(DoubleVisibleConfig(v), lt);
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    return density_matrix_.InitLookup(DoubleVisibleConfig(v), lt);
  }

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    VisibleChangeInfo d_changes =
        DoubleVisibleChangeInfo(tochange, newconf, v.size());
    return density_matrix_.UpdateLookup(DoubleVisibleConfig(v), d_changes.first,
                                        d_changes.second, lt);
  }

  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    auto tochange_d = std::vector<std::vector<int>>(tochange.size());
    auto newconf_d = std::vector<std::vector<double>>(newconf.size());
    // double every element in tochange and newconf
    {
      auto tc = tochange.begin();
      auto nc = newconf.begin();
      int i = 0;
      while (tc != tochange.end()) {
        auto d_changes = DoubleVisibleChangeInfo(*tc, *nc, v.size());
        tochange_d[i] = d_changes.first;
        newconf_d[i] = d_changes.second;
      }
    }
    return density_matrix_.LogValDiff(DoubleVisibleConfig(v), tochange_d,
                                      newconf_d);
  }

  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override {
    VisibleChangeInfo d_changes =
        DoubleVisibleChangeInfo(tochange, newconf, v.size());
    return density_matrix_.LogValDiff(DoubleVisibleConfig(v), d_changes.first,
                                      d_changes.second, lt);
  }

  VectorType DerLog(VisibleConstType v) override {
    return density_matrix_.DerLog(DoubleVisibleConfig(v));
  }

  VectorType DerLog(VisibleConstType v, const LookupType &lt) override {
    return density_matrix_.DerLog(DoubleVisibleConfig(v), lt);
  }

  VectorType DerLogChanged(VisibleConstType v, const std::vector<int> &tochange,
                           const std::vector<double> &newconf) override {
    VisibleChangeInfo d_changes =
        DoubleVisibleChangeInfo(tochange, newconf, v.size());
    return density_matrix_.DerLogChanged(DoubleVisibleConfig(v),
                                         d_changes.first, d_changes.second);
  }

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

  bool IsHolomorphic() override { return density_matrix_.IsHolomorphic(); }

  void to_json(json &j) const override { return density_matrix_.to_json(j); }

  void from_json(const json &j) override {
    return density_matrix_.from_json(j);
  }
};
}  // namespace netket

#endif  // NETKET_DIAGONALDENSITYMATRIX_HPP
