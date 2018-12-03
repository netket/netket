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

#ifndef NETKET_OPERATOR_HPP
#define NETKET_OPERATOR_HPP

#include <memory>

#include "Hilbert/abstract_hilbert.hpp"
#include "Utils/json_utils.hpp"
#include "Utils/memory_utils.hpp"
#include "abstract_operator.hpp"
#include "bosonhubbard.hpp"
#include "graph_hamiltonian.hpp"
#include "heisenberg.hpp"
#include "ising.hpp"
#include "local_operator.hpp"
#include "mpark/variant.hpp"

namespace netket {

class Operator : public AbstractOperator {
 public:
  using VariantType = mpark::variant<Ising, BoseHubbard, Heisenberg,
                                     GraphHamiltonian, LocalOperator>;
  using VectorType = typename AbstractOperator::VectorType;
  using VectorRefType = typename AbstractOperator::VectorRefType;
  using VectorConstRefType = typename AbstractOperator::VectorConstRefType;
  using MelType = typename AbstractOperator::MelType;
  using ConnectorsType = typename AbstractOperator::ConnectorsType;
  using NewconfsType = typename AbstractOperator::NewconfsType;

 private:
  VariantType obj_;

 public:
  Operator(VariantType obj) : obj_(obj) {}

  void FindConn(VectorConstRefType v, MelType &mel, ConnectorsType &connectors,
                NewconfsType &newconfs) const override {
    mpark::visit(
        [v, &mel, &connectors, &newconfs](auto &&obj) {
          obj.FindConn(v, mel, connectors, newconfs);
        },
        obj_);
  }

  std::tuple<MelType, ConnectorsType, NewconfsType> GetConn(
      VectorConstRefType v) const override {
    return mpark::visit([v](auto &&obj) { return obj.GetConn(v); }, obj_);
  }

  Hilbert GetHilbert() const override {
    return mpark::visit([](auto &&obj) { return obj.GetHilbert(); }, obj_);
  }
};

}  // namespace netket
#endif
