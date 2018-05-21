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

#ifndef NETKET_JSONHELPER_HPP
#define NETKET_JSONHELPER_HPP

#include "json.hpp"
#include "json_dumps.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace netket {

template <class T> bool FieldExists(const T &pars, std::string field) {
  return pars.count(field) > 0;
}

template <class T> T FieldVal(const T &pars, std::string field) {
  if (!FieldExists(pars, field)) {
    std::cerr << "Field " << field << " is not defined in the input"
              << std::endl;
    std::abort();
  }
  return pars[field];
}

template <class T>
void FieldArray(const T &pars, std::string field, std::vector<int> &arr) {
  if (!FieldExists(pars, field)) {
    std::cerr << "Field " << field << " is not defined in the input"
              << std::endl;
    std::abort();
  }
  arr.resize(pars[field].size());
  for (int i = 0; i < pars[field].size(); i++) {
    arr[i] = pars[field][i];
  }
}

template <class T, class V>
V FieldOrDefaultVal(const T &pars, std::string field, V defval) {
  if (FieldExists(pars, field)) {
    return pars[field];
  } else {
    return defval;
  }
}

json ReadJsonFromFile(std::string filename) {
  json pars;

  std::ifstream filein(filename);
  if (filein.is_open()) {
    filein >> pars;
  } else {
    std::cerr << "Cannot read Json from file: " << filename << std::endl;
    std::abort();
  }
  return pars;
}

} // namespace netket
#endif
