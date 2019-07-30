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

#include <cmath>
#include "Utils/log_cosh.hpp"

namespace netket {
namespace detail {
namespace {
inline double LogCosh(double x) noexcept {
  x = std::abs(x);
  if (x <= 12.0) {
    return std::log(std::cosh(x));
  } else {
    static const auto log2v = std::log(2.0);
    return x - log2v;
  }
}

inline Complex LogCosh(Complex x) noexcept {
  const double xr = x.real();
  const double xi = x.imag();
  Complex res = LogCosh(xr);
  res += std::log(Complex(std::cos(xi), std::tanh(xr) * std::sin(xi)));
  return res;
}
}  // namespace

Complex SumLogCosh_generic(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> bias) noexcept {
  auto total = Complex{0.0, 0.0};
  for (auto i = Index{0}; i < input.size(); ++i) {
    total += LogCosh(input(i) + bias(i));
  }
  return total;
}

Complex SumLogCosh_generic(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>>
        input) noexcept {
  auto total = Complex{0.0, 0.0};
  for (auto i = Index{0}; i < input.size(); ++i) {
    total += LogCosh(input(i));
  }
  return total;
}
}  // namespace detail
}  // namespace netket
