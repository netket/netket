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

#include "Utils/json_utils.hpp"

#include <complex>
#include <fstream>
#include <vector>

#include <Eigen/Core>
#include <json.hpp>

#include "exceptions.hpp"

namespace nlohmann {
template <typename T>
void adl_serializer<std::complex<T>>::to_json(json &js,
                                              const std::complex<T> &p) {
  js = nlohmann::json{p.real(), p.imag()};
}

template <typename T>
void adl_serializer<std::complex<T> /**/>::from_json(const json &js,
                                                     std::complex<T> &p) {
  if (js.is_array()) {
    p = std::complex<T>(js[0].get<T>(), js[1].get<T>());
  } else {
    p = std::complex<T>(js.get<T>(), 0.);
  }
}

template struct adl_serializer<std::complex<float>>;
template struct adl_serializer<std::complex<double>>;
template struct adl_serializer<std::complex<long double>>;
}  // namespace nlohmann

namespace Eigen {
template <class T>
void to_json(nlohmann::json &js, const Matrix<T, Eigen::Dynamic, 1> &v) {
  std::vector<T> temp(v.size());
  for (std::size_t i = 0; i < std::size_t(v.size()); i++) {
    temp[i] = v(i);
  }
  js = nlohmann::json(temp);
}

template <class T>
void to_json(nlohmann::json &js,
             const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &v) {
  std::vector<std::vector<T>> temp(v.rows());
  for (std::size_t i = 0; i < std::size_t(v.rows()); i++) {
    temp[i].resize(v.cols());
    for (std::size_t j = 0; j < std::size_t(v.cols()); j++) {
      temp[i][j] = v(i, j);
    }
  }
  js = nlohmann::json(temp);
}

template <class T>
void from_json(const nlohmann::json &js, Matrix<T, Eigen::Dynamic, 1> &v) {
  std::vector<T> temp = js.get<std::vector<T>>();
  v.resize(temp.size());
  for (std::size_t i = 0; i < temp.size(); i++) {
    v(i) = temp[i];
  }
}

template <class T>
void from_json(const nlohmann::json &js,
               Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &v) {
  std::vector<std::vector<T>> temp = js.get<std::vector<std::vector<T>>>();

  if (temp[0].size() == 0) {
    throw netket::InvalidInputError(
        "Error while loading Eigen Matrix from Json");
  }

  v.resize(temp.size(), temp[0].size());
  for (std::size_t i = 0; i < temp.size(); i++) {
    for (std::size_t j = 0; j < temp[i].size(); j++) {
      if (temp[i].size() != temp[0].size()) {
        throw netket::InvalidInputError(
            "Error while loading Eigen Matrix from Json");
      }
      v(i, j) = temp[i][j];
    }
  }
}

template void to_json(nlohmann::json &,
                      const Matrix<double, Eigen::Dynamic, 1> &);
template void to_json(nlohmann::json &,
                      const Matrix<std::complex<double>, Eigen::Dynamic, 1> &);
template void to_json(nlohmann::json &,
                      const Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &);
template void to_json(
    nlohmann::json &,
    const Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &);

template void from_json(const nlohmann::json &,
                        Matrix<double, Eigen::Dynamic, 1> &);
template void from_json(const nlohmann::json &,
                        Matrix<std::complex<double>, Eigen::Dynamic, 1> &);
template void from_json(const nlohmann::json &,
                        Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &);
template void from_json(
    const nlohmann::json &,
    Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &);
}  // namespace Eigen

namespace netket {

bool FieldExists(const json &pars, const std::string &field) {
  return pars.count(field) > 0;
}

void CheckFieldExists(const json &pars, const std::string &field,
                      const std::string &context) {
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

json FieldVal(const json &pars, const std::string &field,
              const std::string &context) {
  CheckFieldExists(pars, field, context);
  return pars[field];
}

void FieldArray(const json &pars, const std::string &field,
                std::vector<int> &arr, const std::string &context) {
  CheckFieldExists(pars, field, context);
  arr.resize(pars[field].size());
  for (std::size_t i = 0; i < pars[field].size(); i++) {
    arr[i] = pars[field][i];
  }
}

json ReadJsonFromFile(std::string const &filename) {
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
