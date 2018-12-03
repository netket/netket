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

#ifndef NETKET_HILBERT_HPP
#define NETKET_HILBERT_HPP

#include <memory>
#include <set>
#include "Graph/graph.hpp"
#include "Utils/json_utils.hpp"
#include "Utils/parallel_utils.hpp"
#include "abstract_hilbert.hpp"
#include "bosons.hpp"
#include "custom_hilbert.hpp"
#include "mpark/variant.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "qubits.hpp"
#include "spins.hpp"

namespace netket {
class Hilbert : public AbstractHilbert {
 public:
  using VariantType = mpark::variant<Spin, Boson, Qubit, CustomHilbert>;

 private:
  VariantType obj_;

 public:
  Hilbert(VariantType obj) : obj_(obj) {}

  bool IsDiscrete() const override {
    return mpark::visit([](auto &&obj) { return obj.IsDiscrete(); }, obj_);
  }

  int LocalSize() const override {
    return mpark::visit([](auto &&obj) { return obj.LocalSize(); }, obj_);
  }

  int Size() const override {
    return mpark::visit([](auto &&obj) { return obj.Size(); }, obj_);
  }

  std::vector<double> LocalStates() const override {
    return mpark::visit([](auto &&obj) { return obj.LocalStates(); }, obj_);
  }

  void RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                  netket::default_random_engine &rgen) const override {
    mpark::visit([state, &rgen](auto &&obj) { obj.RandomVals(state, rgen); },
                 obj_);
  }

  void UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                  const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const override {
    mpark::visit([v, &tochange, &newconf](
                     auto &&obj) { obj.UpdateConf(v, tochange, newconf); },
                 obj_);
  }

  Graph GetGraph() const override {
    return mpark::visit([](auto &&obj) { return obj.GetGraph(); }, obj_);
  }
};
}  // namespace netket

#endif
