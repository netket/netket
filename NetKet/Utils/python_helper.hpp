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

#ifndef NETKET_PYTHONHELPER_HPP
#define NETKET_PYTHONHELPER_HPP

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "json_helper.hpp"

namespace netket {

bool FieldExists(const pybind11::kwargs& pars, const std::string& field) {
  return pars.contains(pybind11::cast(field));
}

/**
 * Checks whether @param field exists in @param pars and throws an
 * InvalidInputError if not.
 * @param context is used in the error message to help users locate the location
 * of the error.
 *
 * Example usage: CheckFieldExists(pars["Key"], "SubKey", "Key");
 * If SubKey does not exists, this will throw and error with message
 * "Field 'SubKey' (below 'Key') is not defined in the input".
 */
void CheckFieldExists(const pybind11::kwargs& pars, const std::string& field,
                      const std::string& context = "") {
  if (!FieldExists(pars, field)) {
    std::stringstream s;
    s << "Field '" << field << "' ";
    if (context.size() > 0) {
      s << "(below '" << context << "') ";
    }
    s << "is not defined in the input";
    throw InvalidInputError(s.str());
  }
}

template <class Value>
Value FieldVal(const pybind11::kwargs& pars, const std::string& field,
               const std::string& context = "") {
  CheckFieldExists(pars, field, context);
  return pars[pybind11::cast(field)].cast<Value>();
}

template <class Value>
Value FieldOrDefaultVal(const pybind11::kwargs& pars, const std::string& field,
                        Value defval) {
  if (FieldExists(pars, field)) {
    return pars[pybind11::cast(field)].cast<Value>();
  } else {
    return defval;
  }
}

}  // namespace netket
#endif
