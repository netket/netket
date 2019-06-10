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

#ifndef NETKET_JSONUTILS_HPP
#define NETKET_JSONUTILS_HPP

#include <complex>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

namespace nlohmann {
template <typename T>
struct adl_serializer<std::complex<T>> {
  static void to_json(json& js, const std::complex<T>& p);
  static void from_json(const json& js, std::complex<T>& p);
};
}  // namespace nlohmann

namespace Eigen {
template <class T>
void to_json(nlohmann::json& js, const Matrix<T, Eigen::Dynamic, 1>& v);
template <class T>
void to_json(nlohmann::json& js,
             const Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& v);

template <class T>
void from_json(const nlohmann::json& js, Matrix<T, Eigen::Dynamic, 1>& v);
template <class T>
void from_json(const nlohmann::json& js,
               Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& v);
}  // namespace Eigen

namespace netket {

using json = nlohmann::json;

bool FieldExists(const json& pars, const std::string& field);

void CheckFieldExists(const json& pars, const std::string& field,
                      const std::string& context = "");

json FieldVal(const json& pars, const std::string& field,
              const std::string& context = "");

void FieldArray(const json& pars, const std::string& field,
                std::vector<int>& arr, const std::string& context = "");

json ReadJsonFromFile(std::string filename);

template <class Value, class JSON>
Value FieldVal(const JSON& pars, const std::string& field,
               const std::string& context = "") {
  CheckFieldExists(pars, field, context);
  return pars[field].template get<Value>();
}

template <class Value, class JSON>
Value FieldOrDefaultVal(const JSON& pars, const std::string& field,
                        Value defval) {
  if (FieldExists(pars, field)) {
    return pars[field];
  } else {
    return defval;
  }
}

}  // namespace netket

#endif  // NETKET_JSONUTILS_HPP
