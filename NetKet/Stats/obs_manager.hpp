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

#ifndef NETKET_OBS_MANAGER_HPP
#define NETKET_OBS_MANAGER_HPP

#include <mpi.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "Utils/random_utils.hpp"
#include "common_types.hpp"

#include "binning.hpp"
#include "onlinestat.hpp"

namespace netket {

class ObsManager {
  // TODO(C++14/17): Maybe replace with variant (or e.g. mpark::variant)? In
  // C++11 working with variants is (even) less convenient since C++11 does not
  // support templated lambda functions.
  std::map<std::string, Binning<double>> scalar_real_obs_;
  std::map<std::string, Binning<Eigen::VectorXd>> vector_real_obs_;

 public:
  ObsManager() = default;

  inline void Push(const std::string &name, const double &data) {
    scalar_real_obs_[name] << data;
  }

  inline void Push(const std::string &name, const Eigen::VectorXd &data) {
    vector_real_obs_[name] << data;
  }

  inline void Reset(const std::string &name) {
    if (scalar_real_obs_.count(name) > 0) {
      scalar_real_obs_[name].Reset();
    } else if (vector_real_obs_.count(name) > 0) {
      vector_real_obs_[name].Reset();
    }
  }

  std::vector<std::string> Names() const {
    std::vector<std::string> names;
    names.reserve(Size());
    for (const auto &kv : scalar_real_obs_) {
      names.push_back(kv.first);
    }
    for (const auto &kv : vector_real_obs_) {
      names.push_back(kv.first);
    }
    return names;
  }

  Index Size() const {
    return scalar_real_obs_.size() + vector_real_obs_.size();
  }

  bool Contains(const std::string &name) {
    return scalar_real_obs_.count(name) > 0 || vector_real_obs_.count(name) > 0;
  }

  template <class Map>
  void InsertAllStats(const std::string &name, Map &dict) const {
    if (scalar_real_obs_.count(name) > 0) {
      scalar_real_obs_.at(name).InsertAllStats(dict);
    } else if (vector_real_obs_.count(name) > 0) {
      vector_real_obs_.at(name).InsertAllStats(dict);
    }
  }

  template <class Map>
  void InsertAllStats(Map &dict) const {
    for (const auto &kv : scalar_real_obs_) {
      Map subdict;
      kv.second.InsertAllStats(subdict);
      dict[kv.first.c_str()] = subdict;
    }
    for (const auto &kv : vector_real_obs_) {
      Map subdict;
      kv.second.InsertAllStats(subdict);
      dict[kv.first.c_str()] = subdict;
    }
  }

  json AllStatsJson(const std::string &name) const {
    json j;
    InsertAllStats(name, j);
    return j;
  }
};

void to_json(json &j, const ObsManager &om) {
  auto names = om.Names();
  j = json();
  for (auto name : names) {
    j[name] = om.AllStatsJson(name);
  }
}

}  // namespace netket
#endif
