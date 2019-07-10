// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#ifndef SOURCES_UTILS_LOG_COSH_HPP
#define SOURCES_UTILS_LOG_COSH_HPP

#include <Eigen/Core>

#include "common_types.hpp"

namespace netket {

__attribute__((target("default"))) Complex SumLogCosh(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> bias);

__attribute__((target("avx2"))) Complex SumLogCosh(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> bias);

__attribute__((target("default"))) Complex SumLogCosh(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input);

__attribute__((target("avx2"))) Complex SumLogCosh(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input);

}  // namespace netket

#endif  // SOURCES_UTILS_LOG_COSH_HPP
