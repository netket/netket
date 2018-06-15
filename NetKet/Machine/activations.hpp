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

class AbstractActivation {
 public:
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

  virtual inline void activate(const VectorType &Z, VectorType &A) = 0;
  virtual inline void apply_jacobian(const VectorType &Z, const VectorType &A,
                                     const VectorType &F, VectorType &G) = 0;
  virtual ~AbstractActivation() {}
};

class Identity : public AbstractActivation {
 private:
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

 public:
  // a = activation(z) = z
  // Z = [z1, ..., zn], A = [a1, ..., an], n observations
  inline void activate(const VectorType &Z, VectorType &A) { A.noalias() = Z; }

  // Apply the Jacobian matrix J to a vector f
  // J = d_a / d_z = I
  // g = J * f = f
  // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
  // Note: When entering this function, Z and G may point to the same matrix
  inline void apply_jacobian(const VectorType &Z, const VectorType &A,
                             const VectorType &F, VectorType &G) {
    G.noalias() = F;
    (void)A;
    (void)Z;
  }
};

class Lncosh : public AbstractActivation {
 private:
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

 public:
  // a = activation(z) = z
  // Z = [z1, ..., zn], A = [a1, ..., an], n observations
  inline void activate(const VectorType &Z, VectorType &A) {
    for (int i = 0; i < A.size(); ++i) {
      A(i) = std::log(std::cosh(Z(i)));
    }
  }

  // Apply the Jacobian matrix J to a vector f
  // J = d_a / d_z = I
  // g = J * f = f
  // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
  // Note: When entering this function, Z and G may point to the same matrix
  inline void apply_jacobian(const VectorType &Z, const VectorType &A,
                             const VectorType &F, VectorType &G) {
    G.array() = F.array() * Z.array().tanh();
    (void)A;
  }
};

}  // namespace netket

#endif
