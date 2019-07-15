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

#include "Utils/log_cosh.hpp"

#ifdef NETKET_USE_SLEEF
#include <immintrin.h>
#include <sleef.h>
#endif

#include <cmath>

namespace netket {
namespace {
#ifdef NETKET_USE_SLEEF
inline std::pair<__m256d, __m256d> clog(__m256d const x,
                                        __m256d const y) noexcept {
  auto real = Sleef_logd4_u35avx2(
      _mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)));
  real = _mm256_mul_pd(_mm256_set1_pd(0.5), real);
  auto imag = Sleef_atan2d4_u35avx2(y, x);
  return {real, imag};
}

inline std::pair<__m256d, __m256d> LogCosh(__m256d x, __m256d y) noexcept {
  const auto log_of_2 = _mm256_set1_pd(0.6931471805599453);
  const auto mask = _mm256_cmp_pd(x, _mm256_set1_pd(0), _CMP_LT_OQ);
  x = _mm256_blendv_pd(x, _mm256_sub_pd(_mm256_set1_pd(0), x), mask);
  y = _mm256_blendv_pd(y, _mm256_sub_pd(_mm256_set1_pd(0), y), mask);
  const auto exp_min_2x =
      Sleef_expd4_u10avx2(_mm256_mul_pd(_mm256_set1_pd(-2.0), x));
  const auto _t = Sleef_sincosd4_u35avx2(y);
  auto p = _t.y;
  auto q = _t.x;
  p = _mm256_mul_pd(p, _mm256_add_pd(_mm256_set1_pd(1.0), exp_min_2x));
  q = _mm256_mul_pd(q, _mm256_sub_pd(_mm256_set1_pd(1.0), exp_min_2x));
  std::tie(p, q) = clog(p, q);
  p = _mm256_sub_pd(x, _mm256_sub_pd(log_of_2, p));
  return {p, q};
}

inline std::pair<__m256d, __m256d> Load(const Complex* data) noexcept {
  static_assert(sizeof(__m256d) == 2 * sizeof(Complex), "");
  const auto* p = reinterpret_cast<const double*>(data);
  return {_mm256_loadu_pd(p), _mm256_loadu_pd(p + 4)};
}

inline Complex ToComplex(__m128d z) noexcept {
  static_assert(sizeof(__m128d) == sizeof(Complex), "");
  alignas(16) Complex r;
  _mm_storeu_pd(reinterpret_cast<double*>(&r), z);
  return r;
}

inline __m256d SumLogCoshKernel(__m256d z1, __m256d z2) noexcept {
  auto x = _mm256_unpacklo_pd(z1, z2);
  auto y = _mm256_unpackhi_pd(z1, z2);
  std::tie(x, y) = LogCosh(x, y);
  z1 = _mm256_shuffle_pd(x, y, 0b0000);
  z2 = _mm256_shuffle_pd(x, y, 0b1111);
  return _mm256_add_pd(z1, z2);
}

Complex SumLogCosh(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    std::true_type /*has AVX2*/) noexcept {
  constexpr auto vector_size =
      static_cast<Index>(sizeof(__m256d) / sizeof(double));
  static_assert(vector_size == 4, "");

  auto total = _mm256_set1_pd(0.0);
  auto n = static_cast<int64_t>(input.size());
  const auto* input_ptr = reinterpret_cast<const double*>(input.data());
  for (; n >= vector_size; n -= vector_size, input_ptr += 2 * vector_size) {
    auto z1 = _mm256_loadu_pd(input_ptr);
    auto z2 = _mm256_loadu_pd(input_ptr + vector_size);
    total = _mm256_add_pd(total, SumLogCoshKernel(z1, z2));
  }
  if (n != 0) {
    alignas(32) double temp[2 * vector_size] = {};
    assert(std::all_of(temp, temp + 2 * vector_size,
                       [](double x) { return x == 0.0; }));
    for (auto i = Index{0}; i < 2 * n; ++i) {
      temp[i] = input_ptr[i];
    }
    auto z1 = _mm256_loadu_pd(temp);
    auto z2 = _mm256_loadu_pd(temp + vector_size);
    total = _mm256_add_pd(total, SumLogCoshKernel(z1, z2));
  }
  return ToComplex(_mm_add_pd(_mm256_extractf128_pd(total, 0),
                              _mm256_extractf128_pd(total, 1)));
}

Complex SumLogCosh(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> bias,
    std::true_type /*has AVX2*/) noexcept {
  assert(input.size() == bias.size() && "incompatible sizes");
  constexpr auto vector_size =
      static_cast<Index>(sizeof(__m256d) / sizeof(double));
  static_assert(vector_size == 4, "");

  auto total = _mm256_set1_pd(0.0);
  auto n = static_cast<int64_t>(input.size());
  const auto* input_ptr = reinterpret_cast<const double*>(input.data());
  const auto* bias_ptr = reinterpret_cast<const double*>(bias.data());
  for (; n >= vector_size; n -= vector_size, input_ptr += 2 * vector_size,
                           bias_ptr += 2 * vector_size) {
    auto z1 = _mm256_loadu_pd(input_ptr);
    auto z2 = _mm256_loadu_pd(input_ptr + vector_size);
    z1 = _mm256_add_pd(z1, _mm256_loadu_pd(bias_ptr));
    z2 = _mm256_add_pd(z2, _mm256_loadu_pd(bias_ptr + vector_size));
    total = _mm256_add_pd(total, SumLogCoshKernel(z1, z2));
  }
  if (n != 0) {
    alignas(32) double temp[2 * vector_size] = {};
    assert(std::all_of(temp, temp + 2 * vector_size,
                       [](double x) { return x == 0.0; }));
    for (auto i = Index{0}; i < 2 * n; ++i) {
      temp[i] = input_ptr[i] + bias_ptr[i];
    }
    auto z1 = _mm256_loadu_pd(temp);
    auto z2 = _mm256_loadu_pd(temp + vector_size);
    total = _mm256_add_pd(total, SumLogCoshKernel(z1, z2));
  }
  return ToComplex(_mm_add_pd(_mm256_extractf128_pd(total, 0),
                              _mm256_extractf128_pd(total, 1)));
}
#endif  // NETKET_USE_SLEEF

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

Complex SumLogCosh(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> bias,
    std::false_type /*has AVX2*/) noexcept {
  auto total = Complex{0.0, 0.0};
  for (auto i = Index{0}; i < input.size(); ++i) {
    total += LogCosh(input(i) + bias(i));
  }
  return total;
}

Complex SumLogCosh(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    std::false_type /*has AVX2*/) noexcept {
  auto total = Complex{0.0, 0.0};
  for (auto i = Index{0}; i < input.size(); ++i) {
    total += LogCosh(input(i));
  }
  return total;
}
}  // namespace

Complex SumLogCosh(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> bias) noexcept {
#ifdef NETKET_USE_SLEEF
  if (__builtin_cpu_supports("avx2")) {
    return SumLogCosh(input, bias, std::true_type{});
  } else {
    return SumLogCosh(input, bias, std::false_type{});
  }
#else
  return SumLogCosh(input, bias, std::false_type{});
#endif
}

Complex SumLogCosh(Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>>
                       input) noexcept {
#ifdef NETKET_USE_SLEEF
  if (__builtin_cpu_supports("avx2")) {
    return SumLogCosh(input, std::true_type{});
  } else {
    return SumLogCosh(input, std::false_type{});
  }
#else
  return SumLogCosh(input, std::false_type{});
#endif
}

}  // namespace netket
