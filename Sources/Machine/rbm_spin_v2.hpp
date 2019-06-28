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

#ifndef SOURCES_MACHINE_RBM_SPIN_V2_HPP
#define SOURCES_MACHINE_RBM_SPIN_V2_HPP

#include <cmath>

#include <nonstd/optional.hpp>

#include "Machine/abstract_machine.hpp"

#include <immintrin.h>
#include <sleef.h>

namespace netket {

inline std::pair<__m256d, __m256d> clog(__m256d const x,
                                        __m256d const y) noexcept {
  auto real = Sleef_logd4_u10avx2(
      _mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)));
  real = _mm256_mul_pd(_mm256_set1_pd(0.5), real);
  auto imag = Sleef_atan2d4_u10avx2(y, x);
  return {real, imag};
}

inline std::pair<__m256d, __m256d> LogCosh(__m256d x, __m256d y) noexcept {
  const auto log_of_2 = _mm256_set1_pd(0.6931471805599453);
  const auto mask = _mm256_cmp_pd(x, _mm256_set1_pd(0), _CMP_LT_OQ);
  x = _mm256_blendv_pd(x, _mm256_sub_pd(_mm256_set1_pd(0), x), mask);
  y = _mm256_blendv_pd(y, _mm256_sub_pd(_mm256_set1_pd(0), y), mask);
  const auto exp_min_2x =
      Sleef_expd4_u10avx2(_mm256_mul_pd(_mm256_set1_pd(-2.0), x));
  const auto _t = Sleef_sincosd4_u10avx2(y);
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

inline Complex SumLogCosh(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> bias) {
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

inline Complex SumLogCoshDumb(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> bias) {
  auto total = Complex{0.0, 0.0};
  for (auto i = Index{0}; i < input.size(); ++i) {
    total += std::log(std::cosh(input(i) + bias(i)));
  }
  return total;
}

inline Complex clog(Complex z) {
  return Complex{0.5 * std::log(z.real() * z.real() + z.imag() * z.imag()),
                 std::atan2(z.imag(), z.real())};
}

namespace detail {
inline Complex LogCosh(double x, double y) {
  constexpr const auto cutoff = 12.0;
  constexpr const auto log_of_2 = 0.6931471805599453;
  if (x < 0.0) {
    x = -x;
    y = -y;
  }
  if (x > cutoff) {
    constexpr const auto pi = 3.141592653589793;
    constexpr const auto two_pi = 6.2831853071795865;
    y = std::fmod(y, two_pi);
    if (y > pi) {
      y -= two_pi;
    } else if (y < -pi) {
      y += two_pi;
    }
    return Complex{x - log_of_2, y};
  }
  const auto exp_min_2x = std::exp(-2.0 * x);
  const auto sin_y = std::sin(y);
  const auto cos_y = std::cos(y);
  const auto t =
      std::log(Complex{cos_y + cos_y * exp_min_2x, sin_y * (1.0 - exp_min_2x)});
  return Complex{x - (log_of_2 - t.real()), t.imag()};
}

inline Complex LogCosh(const Complex z) { return LogCosh(z.real(), z.imag()); }

inline double lncosh(double x) {
  const double xp = std::abs(x);
  if (xp <= 12.) {
    return std::log(std::cosh(xp));
  } else {
    static const auto log2v = std::log(2.0);
    return xp - log2v;
  }
}

inline Complex lncosh(Complex x) {
  const double xr = x.real();
  const double xi = x.imag();

  Complex res = lncosh(xr);
  res += std::log(Complex(std::cos(xi), std::tanh(xr) * std::sin(xi)));

  return res;
}
}  // namespace detail

class RbmSpinV2 {
  template <class T>
  using M = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  template <class T>
  using V = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  static constexpr auto D = Eigen::Dynamic;

  Eigen::Matrix<Complex, D, D, Eigen::ColMajor> W_;   ///< weights
  nonstd::optional<Eigen::Matrix<Complex, D, 1>> a_;  ///< visible units bias
  nonstd::optional<Eigen::Matrix<Complex, D, 1>> b_;  ///< hidden units bias

  /// Caches
  Eigen::Matrix<Complex, D, D, Eigen::RowMajor> theta_;
  Eigen::Matrix<Complex, D, 1> output_;

  /// Hilbert space
  std::shared_ptr<const AbstractHilbert> hilbert_;

 public:
  RbmSpinV2(std::shared_ptr<const AbstractHilbert> hilbert, Index nhidden,
            Index alpha, bool usea, bool useb, Index const batch_size)
      : W_{}, a_{nonstd::nullopt}, b_{nonstd::nullopt}, theta_{}, output_{} {
    const auto nvisible = hilbert->Size();
    assert(nvisible >= 0 && "AbstractHilbert::Size is broken");
    if (nhidden < 0) {
      std::ostringstream msg;
      msg << "invalid number of hidden units: " << nhidden
          << "; expected a non-negative number";
      throw InvalidInputError{msg.str()};
    }
    if (alpha < 0) {
      std::ostringstream msg;
      msg << "invalid density of hidden units: " << alpha
          << "; expected a non-negative number";
      throw InvalidInputError{msg.str()};
    }
    if (nhidden > 0 && alpha > 0 && nhidden != alpha * nvisible) {
      std::ostringstream msg;
      msg << "number and density of hidden units are incompatible: " << nhidden
          << " != " << alpha << " * " << nvisible;
      throw InvalidInputError{msg.str()};
    }
    nhidden = std::max(nhidden, alpha * nvisible);

    W_.resize(nvisible, nhidden);
    if (usea) {
      a_.emplace(nvisible);
    }
    if (useb) {
      b_.emplace(nhidden);
    }

    theta_.resize(batch_size, nhidden);
    output_.resize(batch_size);
  }

  Index Nvisible() const noexcept { return W_.rows(); }
  Index Nhidden() const noexcept { return W_.cols(); }
  Index Npar() const noexcept {
    return W_.size() + (a_.has_value() ? a_->size() : 0) +
           (b_.has_value() ? b_->size() : 0);
  }
  Index BatchSize() const noexcept { return theta_.rows(); }

  Eigen::Ref<Eigen::Matrix<Complex, D, 1> const> LogVal(
      Eigen::Ref<Eigen::Matrix<double, D, D, Eigen::RowMajor> const> x) {
    if (x.rows() != BatchSize() || x.cols() != Nvisible()) {
      std::ostringstream msg;
      msg << "wrong shape: [" << x.rows() << ", " << x.cols() << "]; expected ["
          << BatchSize() << ", " << Nvisible() << "]\n";
      throw InvalidInputError{msg.str()};
    }
    if (a_.has_value()) {
      output_.noalias() = x * (*a_);
    } else {
      output_.setZero();
    }
    theta_.noalias() = x * W_;
    ApplyBiasAndActivation();
    return output_;
  }

  Eigen::Ref<Eigen::Matrix<Complex, D, D>> GetW() { return W_; }
  Eigen::Ref<Eigen::Matrix<Complex, D, 1>> GetA() { return a_.value(); }
  Eigen::Ref<Eigen::Matrix<Complex, D, 1>> GetB() { return b_.value(); }

  void ApplyBiasAndActivation() {
    if (b_.has_value()) {
#pragma omp parallel for
      for (auto j = Index{0}; j < BatchSize(); ++j) {
        output_(j) += SumLogCosh(theta_.row(j), (*b_));  // total;
#if 0
        Complex total{0.0, 0.0};
        for (auto i = Index{0}; i < Nhidden(); ++i) {
          const auto bias = (*b_)(i);
          total += detail::lncosh(
              theta_(j, i) + bias);  // detail::LogCosh(theta_(j, i) + bias);
        }
        output_(j) += total;
#endif
      }
    } else {
      for (auto i = Index{0}; i < Nhidden(); ++i) {
        for (auto j = Index{0}; j < BatchSize(); ++j) {
          output_(j) += detail::LogCosh(theta_(j, i));
        }
      }
    }
  }

  // PyObject *StateDict() const;
  // void StateDict(PyObject *dict);
};

}  // namespace netket

#endif  // SOURCES_MACHINE_RBM_SPIN_V2_HPP
