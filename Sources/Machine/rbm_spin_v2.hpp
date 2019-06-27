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

namespace netket {

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

double lncosh(double x) {
  const double xp = std::abs(x);
  if (xp <= 12.) {
    return std::log(std::cosh(xp));
  } else {
    static const auto log2v = std::log(2.0);
    return xp - log2v;
  }
}

Complex lncosh(Complex x) {
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
  Eigen::Matrix<Complex, D, D, Eigen::ColMajor> theta_;
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
      Eigen::Ref<Eigen::Matrix<Complex, D, D, Eigen::RowMajor> const> x) {
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
    theta_.noalias() = W_ * x;
    ApplyBiasAndActivation();
    return output_;
  }

  Eigen::Ref<Eigen::Matrix<Complex, D, D>> GetW() { return W_; }
  Eigen::Ref<Eigen::Matrix<Complex, D, 1>> GetA() { return a_.value(); }
  Eigen::Ref<Eigen::Matrix<Complex, D, 1>> GetB() { return b_.value(); }

  void ApplyBiasAndActivation() {
    if (b_.has_value()) {
      for (auto i = Index{0}; i < Nhidden(); ++i) {
        const auto bias = (*b_)(i);
        for (auto j = Index{0}; j < BatchSize(); ++j) {
          output_(j) += detail::LogCosh(theta_(j, i) + bias);
        }
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
