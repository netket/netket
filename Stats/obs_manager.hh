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

#ifndef NETKET_OBS_MANAGER_HH
#define NETKET_OBS_MANAGER_HH

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include <map>
#include <mpi.h>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace netket {

class ObsManager {

  std::map<std::string, Binning<double>> scalar_real_obs_;
  std::map<std::string, Binning<Eigen::VectorXd>> vector_real_obs_;

public:
  ObsManager() {}
  inline void Push(std::string name, const double &data) {
    scalar_real_obs_[name] << data;
  }

  inline void Push(std::string name, const Eigen::VectorXd &data) {
    vector_real_obs_[name] << data;
  }

  inline void Reset(std::string name) {
    if (scalar_real_obs_.count(name) > 0) {
      scalar_real_obs_[name].Reset();
    } else if (vector_real_obs_.count(name) > 0) {
      vector_real_obs_[name].Reset();
    }
  }

  std::vector<std::string> Names() const {
    std::vector<std::string> names;
    for (auto it = scalar_real_obs_.begin(); it != scalar_real_obs_.end();
         ++it) {
      names.push_back(it->first);
    }
    for (auto it = vector_real_obs_.begin(); it != vector_real_obs_.end();
         ++it) {
      names.push_back(it->first);
    }
    return names;
  }

  json AllStats(std::string name) const {
    json j;
    if (scalar_real_obs_.count(name) > 0) {
      j = scalar_real_obs_.at(name).AllStats();
    } else if (vector_real_obs_.count(name) > 0) {
      j = vector_real_obs_.at(name).AllStats();
    }
    return j;
  }
};

void to_json(json &j, const ObsManager &om) {
  auto names = om.Names();
  j = json();
  for (auto name : names) {
    j[name] = om.AllStats(name);
  }
}

} // namespace netket
#endif
