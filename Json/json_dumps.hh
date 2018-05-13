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

#ifndef NETKET_JSON_DUMPS_HH
#define NETKET_JSON_DUMPS_HH

#include "Json/json.hpp"
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <vector>

using json = nlohmann::json;

namespace Eigen {

template <class T>
void to_json(json &js, const Matrix<T, Eigen::Dynamic, 1> &v) {
  std::vector<T> temp(v.size());
  for (std::size_t i = 0; i < std::size_t(v.size()); i++) {
    temp[i] = v(i);
  }
  js = json(temp);
}

template <class T>
void from_json(const json &js, Matrix<T, Eigen::Dynamic, 1> &v) {
  std::vector<T> temp = js.get<std::vector<T>>();
  v.resize(temp.size());
  for (std::size_t i = 0; i < temp.size(); i++) {
    v(i) = temp[i];
  }
}

template <class T>
void to_json(json &js, const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &v) {
  std::vector<std::vector<T>> temp(v.rows());
  for (std::size_t i = 0; i < std::size_t(v.rows()); i++) {
    temp[i].resize(v.cols());
    for (std::size_t j = 0; j < std::size_t(v.cols()); j++) {
      temp[i][j] = v(i, j);
    }
  }
  js = json(temp);
}

template <class T>
void from_json(const json &js, Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &v) {
  std::vector<std::vector<T>> temp = js.get<std::vector<std::vector<T>>>();

  if (temp[0].size() == 0) {
    std::cerr << "Error while loading Eigen Matrix from Json" << std::endl;
    std::abort();
  }

  v.resize(temp.size(), temp[0].size());
  for (std::size_t i = 0; i < temp.size(); i++) {
    for (std::size_t j = 0; j < temp[i].size(); j++) {
      if (temp[i].size() != temp[0].size()) {
        std::cerr << "Error while loading Eigen Matrix from Json" << std::endl;
        std::abort();
      }
      v(i, j) = temp[i][j];
    }
  }
}

} // namespace Eigen

namespace std {

void to_json(json &js, const std::complex<double> &p) {
  js = json{p.real(), p.imag()};
}

void from_json(const json &js, std::complex<double> &p) {
  if (js.is_array()) {
    p = std::complex<double>(js[0].get<double>(), js[1].get<double>());
  } else {
    p = std::complex<double>(js.get<double>(), 0.);
  }
}

} // namespace std

#endif
