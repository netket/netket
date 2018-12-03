// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
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

#ifndef NETKET_MACHINE_HPP
#define NETKET_MACHINE_HPP

#include <fstream>
#include <memory>

#include "Graph/graph.hpp"
#include "Operator/operator.hpp"
#include "abstract_machine.hpp"
#include "ffnn.hpp"
#include "jastrow.hpp"
#include "jastrow_symm.hpp"
#include "mpark/variant.hpp"
#include "mps_periodic.hpp"
#include "rbm_multival.hpp"
#include "rbm_spin.hpp"
#include "rbm_spin_symm.hpp"

namespace netket {

template <typename T>
class Machine : public AbstractMachine<T> {
 public:
  using VariantType =
      mpark::variant<RbmSpin<T>, RbmSpinSymm<T>, FFNN<T>, Jastrow<T>,
                     JastrowSymm<T>, RbmMultival<T>, MPSPeriodic<T, true>,
                     MPSPeriodic<T, false>>;
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;
  using VectorRefType = typename AbstractMachine<T>::VectorRefType;
  using VectorConstRefType = typename AbstractMachine<T>::VectorConstRefType;
  using VisibleConstType = typename AbstractMachine<T>::VisibleConstType;

 private:
  VariantType obj_;

 public:
  Machine(VariantType obj) : obj_(std::move(obj)) {}

  int Npar() const override {
    return mpark::visit([](auto &&obj) { return obj.Npar(); }, obj_);
  }
  VectorType GetParameters() override {
    return mpark::visit([](auto &&obj) { return obj.GetParameters(); }, obj_);
  }
  void SetParameters(VectorConstRefType pars) override {
    mpark::visit([pars](auto &&obj) { obj.SetParameters(pars); }, obj_);
  }
  void InitRandomPars(int seed, double sigma) override {
    mpark::visit([=](auto &&obj) { obj.InitRandomPars(seed, sigma); }, obj_);
  }
  int Nvisible() const override {
    return mpark::visit([](auto &&obj) { return obj.Nvisible(); }, obj_);
  }
  T LogVal(VisibleConstType v) override {
    return mpark::visit([=](auto &&obj) { return obj.LogVal(v); }, obj_);
  }
  T LogVal(VisibleConstType v, const LookupType &lt) override {
    return mpark::visit([v, &lt](auto &&obj) { return obj.LogVal(v, lt); },
                        obj_);
  }
  void InitLookup(VisibleConstType v, LookupType &lt) override {
    mpark::visit([v, &lt](auto &&obj) { obj.InitLookup(v, lt); }, obj_);
  }
  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    mpark::visit(
        [v, &tochange, &newconf, &lt](auto &&obj) {
          return obj.UpdateLookup(v, tochange, newconf, lt);
        },
        obj_);
  }
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    return mpark::visit(
        [v, &tochange, &newconf](auto &&obj) {
          return obj.LogValDiff(v, tochange, newconf);
        },
        obj_);
  }
  T LogValDiff(VisibleConstType v, const std::vector<int> &toflip,
               const std::vector<double> &newconf,
               const LookupType &lt) override {
    return mpark::visit(
        [v, &toflip, &newconf, &lt](auto &&obj) {
          return obj.LogValDiff(v, toflip, newconf, lt);
        },
        obj_);
  }
  VectorType DerLog(VisibleConstType v) override {
    return mpark::visit([=](auto &&obj) { return obj.DerLog(v); }, obj_);
  }

  void to_json(json &j) const override {
    mpark::visit([&j](auto &&obj) { obj.to_json(j); }, obj_);
  }
  void from_json(const json &j) override {
    mpark::visit([&j](auto &&obj) { obj.from_json(j); }, obj_);
  }

  Hilbert GetHilbert() const override {
    return mpark::visit([](auto &&obj) { return obj.GetHilbert(); }, obj_);
  }
};

}  // namespace netket
#endif
