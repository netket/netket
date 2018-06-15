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
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

  virtual inline void Activate(const VectorType &Z, VectorType &A) = 0;
  virtual inline void ApplyJacobian(const VectorType &Z, const VectorType &A,
                                    const VectorType &F, VectorType &G) = 0;
  virtual ~AbstractActivation() {}
};

class Identity : public AbstractActivation {
 private:
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

 public:
  // A = Z
  inline void Activate(const VectorType &Z, VectorType &A) { A.noalias() = Z; }

  // Apply the (derivative of activation function) matrix J to a vector F
  // A = Z
  // J = dA / dZ = I
  // G = J * F = F
  inline void ApplyJacobian(const VectorType & /*Z*/, const VectorType & /*A*/,
                            const VectorType &F, VectorType &G) {
    G.noalias() = F;
  }
};

class Lncosh : public AbstractActivation {
 private:
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

 public:
  // A = Lncosh(Z)
  inline void Activate(const VectorType &Z, VectorType &A) {
    for (int i = 0; i < A.size(); ++i) {
      A(i) = std::log(std::cosh(Z(i)));
    }
  }

  // Apply the (derivative of activation function) matrix J to a vector F
  // A = Lncosh(Z)
  // J = dA / dZ
  // G = J * F
  inline void ApplyJacobian(const VectorType &Z, const VectorType & /*A*/,
                            const VectorType &F, VectorType &G) {
    G.array() = F.array() * Z.array().tanh();
  }
};

}  // namespace netket

#endif
