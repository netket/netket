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
class Hilbert {
 public:
  using VariantType = mpark::variant<Spin, Boson, Qubit, CustomHilbert>;

 private:
  VariantType obj_;

 public:
  Hilbert(VariantType obj) : obj_(std::move(obj)) {}

  /**
  Member function returning true if the hilbert space has discrete quantum
  numbers.
  @return true if the local hilbert space is discrete
  */
  bool IsDiscrete() const {
    return mpark::visit([](auto &&obj) { return obj.IsDiscrete(); }, obj_);
  }

  /**
  Member function returning the size of the local hilbert space.
  @return Size of the discrete local hilbert space. For continous spaces an
  error message is returned.
  */
  int LocalSize() const {
    return mpark::visit([](auto &&obj) { return obj.LocalSize(); }, obj_);
  }
  /**
  Member function returning the number of visible units needed to describe the
  system.
  @return Number of visible units needed to described the system.
  */
  int Size() const {
    return mpark::visit([](auto &&obj) { return obj.Size(); }, obj_);
  }

  /**
  Member function returning the local states.
  @return Vector containing the value of the discrete local quantum numbers. If
  the local quantum numbers are continous, the vector contains lower and higher
  bounds for the local quantum numbers.
  */
  std::vector<double> LocalStates() const {
    return mpark::visit([](auto &&obj) { return obj.LocalStates(); }, obj_);
  }

  /**
  Member function generating uniformely distributed local random states
  @param state a reference to a visible configuration, in output this contains
  the random state.
  @param rgen the random number generator to be used
  */
  void RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                  netket::default_random_engine &rgen) const {
    mpark::visit([state, &rgen](auto &&obj) { obj.RandomVals(state, rgen); },
                 obj_);
  }

  /**
  Member function updating a visible configuration using the information on
  where the local changes have been done.
  @param v is the vector visible units to be modified.
  @param tochange contains a list of which quantum numbers are to be
  modified.
  @param newconf contains the value that those quantum numbers should take
  */
  void UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                  const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const {
    mpark::visit([v, &tochange, &newconf](
                     auto &&obj) { obj.UpdateConf(v, tochange, newconf); },
                 obj_);
  }

  Graph GetGraph() const {
    return mpark::visit([](auto &&obj) { return obj.GetGraph(); }, obj_);
  }
};
}  // namespace netket

#endif
