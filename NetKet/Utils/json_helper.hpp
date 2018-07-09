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

#include <fstream>
#include <iostream>
#include <json.hpp>
#include <string>
#include <vector>

#include "exceptions.hpp"

namespace netket {

using json = nlohmann::json;

template<class Json>
bool FieldExists(const Json &pars, const std::string& field)
{
  return pars.count(field) > 0;
}

/**
 * Checks whether @param field exists in @param pars and throws an InvalidInputError if not.
 * @param context is used in the error message to help users locate the location of the error.
 *
 * Example usage: CheckFieldExists(pars["Key"], "SubKey", "Key");
 * If SubKey does not exists, this will throw and error with message
 * "Field 'SubKey' (below 'Key') is not defined in the input".
 */
template<class Json>
void CheckFieldExists(const Json& pars, const std::string& field, const std::string& context = "")
{
  if(!FieldExists(pars, field)) {
    std::stringstream s;
    s << "Field '" << field << "' ";
    if(context.size() > 0) {
        s << "(below '" << context << "') ";
    }
    s << "is not defined in the input";
    throw InvalidInputError(s.str());
  }
}

template<class Json>
Json FieldVal(const Json &pars, const std::string& field, const std::string& context = "") {
  CheckFieldExists(pars, field, context);
  return pars[field];
}

template<class Json>
void FieldArray(const Json &pars, const std::string& field, std::vector<int> &arr,
                const std::string& context = "") {
  CheckFieldExists(pars, field, context);
  arr.resize(pars[field].size());
  for (int i = 0; i < pars[field].size(); i++) {
    arr[i] = pars[field][i];
  }
}

template <class Json, class Value>
Value FieldOrDefaultVal(const Json &pars, std::string field, Value defval) {
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
    std::stringstream s;
    s << "Cannot read Json from file: " << filename;
    throw InvalidInputError(s.str());
  }
  return pars;
}

}  // namespace netket
#endif
