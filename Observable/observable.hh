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

#ifndef NETKET_OBSERVABLE_HH
#define NETKET_OBSERVABLE_HH

#include <string>
#include <vector>

#include "abstract_observable.hh"
#include "custom_observable.hh"

namespace netket {

class Observable : public AbstractObservable {

  using Ptype = std::unique_ptr<AbstractObservable>;
  Ptype o_;

public:
  using MatType = LocalOperator::MatType;

  Observable(const Hilbert &hilbert, const json &obspars) {

    if (!FieldExists(obspars, "Operators")) {
      std::cerr << "Observable's Operators not defined" << std::endl;
      std::abort();
    }
    if (!FieldExists(obspars, "ActingOn")) {
      std::cerr << "Observable's ActingOn not defined" << std::endl;
      std::abort();
    }
    if (!FieldExists(obspars, "Name")) {
      std::cerr << "Observable's Name not defined" << std::endl;
      std::abort();
    }

    auto jop = obspars.at("Operators").get<std::vector<MatType>>();
    auto sites = obspars.at("ActingOn").get<std::vector<std::vector<int>>>();
    std::string name = obspars.at("Name");

    o_ = Ptype(new CustomObservable(hilbert, jop, sites, name));
  }

  void FindConn(const Eigen::VectorXd &v,
                std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) {
    return o_->FindConn(v, mel, connectors, newconfs);
  }

  const Hilbert &GetHilbert() const { return o_->GetHilbert(); }

  const std::string Name() const { return o_->Name(); }
};
} // namespace netket

#include "observables.hh"

#endif
