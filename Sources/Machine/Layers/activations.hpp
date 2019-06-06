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

#ifndef NETKET_ACTIVATIONS_HPP
#define NETKET_ACTIVATIONS_HPP

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

namespace netket {

/**
  Abstract class for Activations.
*/
class AbstractActivation {
 public:
  using VectorType = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
  using VectorRefType = Eigen::Ref<VectorType>;
  using VectorConstRefType = Eigen::Ref<const VectorType>;

  virtual void operator()(VectorConstRefType Z, VectorRefType A) const = 0;

  // Z is the layer output before applying nonlinear function
  // A = nonlinearfunction(Z)
  // F = dL/dA is the derivative of A wrt the output L = log(psi(v))
  // G is the place to write the output i.e. G = dL/dZ = dL/dA * dA/dZ
  virtual void ApplyJacobian(VectorConstRefType Z, VectorConstRefType A,
                             VectorConstRefType F, VectorRefType G) const = 0;
  virtual ~AbstractActivation() {}
};

inline double lncosh(double x) {
  const double xp = std::abs(x);
  if (xp <= 12.) {
    return std::log(std::cosh(xp));
  } else {
    const static double log2v = std::log(2.);
    return xp - log2v;
  }
}

// ln(cos(x)) for std::complex argument
// the modulus is computed by means of the previously defined function
// for real argument
inline Complex lncosh(Complex x) {
  const double xr = x.real();
  const double xi = x.imag();

  Complex res = lncosh(xr);
  res += std::log(
      Complex(std::cos(xi), std::tanh(xr) * std::sin(xi)));

  return res;
}

class Identity : public AbstractActivation {
  using VectorType = typename AbstractActivation::VectorType;

 public:
  // A = Z
  inline void operator()(VectorConstRefType Z, VectorRefType A) const override {
    A.noalias() = Z;
  }

  // Apply the (derivative of activation function) matrix J to a vector F
  // A = Z
  // J = dA / dZ = I
  // G = J * F = F
  inline void ApplyJacobian(VectorConstRefType /*Z*/, VectorConstRefType /*A*/,
                            VectorConstRefType F,
                            VectorRefType G) const override {
    G.noalias() = F;
  }
};

class Lncosh : public AbstractActivation {
  using VectorType = typename AbstractActivation::VectorType;

 public:
  std::string name = "Lncosh";
  // A = Lncosh(Z)
  inline void operator()(VectorConstRefType Z, VectorRefType A) const override {
    for (int i = 0; i < A.size(); ++i) {
      A(i) = lncosh(Z(i));
    }
  }

  // Apply the (derivative of activation function) matrix J to a vector F
  // A = Lncosh(Z)
  // J = dA / dZ
  // G = J * F
  inline void ApplyJacobian(VectorConstRefType Z, VectorConstRefType /*A*/,
                            VectorConstRefType F,
                            VectorRefType G) const override {
    G.array() = F.array() * Z.array().tanh();
  }
};

class Tanh : public AbstractActivation {
  using VectorType = typename AbstractActivation::VectorType;

 public:
  std::string name = "Tanh";
  // A = Tanh(Z)
  inline void operator()(VectorConstRefType Z, VectorRefType A) const override {
    A.array() = Z.array().tanh();
  }

  // Apply the (derivative of activation function) matrix J to a vector F
  // A = Tanh(Z)
  // J = dA / dZ
  // G = J * F
  inline void ApplyJacobian(VectorConstRefType /*Z*/, VectorConstRefType A,
                            VectorConstRefType F,
                            VectorRefType G) const override {
    G.array() = F.array() * (1 - A.array() * A.array());
  }
};

class Relu : public AbstractActivation {
  using VectorType = typename AbstractActivation::VectorType;

  double theta1_ = std::atan(1) * 3;
  double theta2_ = -std::atan(1);

 public:
  std::string name = "Relu";
  // A = Z
  inline void operator()(VectorConstRefType Z, VectorRefType A) const override {
    for (int i = 0; i < Z.size(); ++i) {
      A(i) =
          (std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? Z(i) : 0.0;
    }
  }

  // Apply the (derivative of activation function) matrix J to a vector F
  // A = Z
  // J = dA / dZ = I
  // G = J * F = F
  inline void ApplyJacobian(VectorConstRefType Z, VectorConstRefType /*A*/,
                            VectorConstRefType F,
                            VectorRefType G) const override {
    for (int i = 0; i < Z.size(); ++i) {
      G(i) =
          (std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? F(i) : 0.0;
    }
  }
};

}  // namespace netket

#endif
