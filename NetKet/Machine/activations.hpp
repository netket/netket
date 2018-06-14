#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

#ifndef NETKET_ACTIVATIONS_HPP
#define NETKET_ACTIVATIONS_HPP

namespace netket {

class AbstractActivation {
public:
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

  virtual inline void activate(const VectorType &Z, VectorType &A) = 0;
  virtual inline void apply_jacobian(const VectorType &Z, const VectorType
  &A,
                                     const VectorType &F, VectorType &G) = 0;
};

class Identity : public AbstractActivation {
private:
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

public:
  // a = activation(z) = z
  // Z = [z1, ..., zn], A = [a1, ..., an], n observations
  inline void activate(const VectorType &Z, VectorType &A) {
    A.noalias() = Z;
  }

  // Apply the Jacobian matrix J to a vector f
  // J = d_a / d_z = I
  // g = J * f = f
  // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
  // Note: When entering this function, Z and G may point to the same matrix
  inline void apply_jacobian(const VectorType &Z, const VectorType &A,
                                    const VectorType &F, VectorType &G) {
    G.noalias() = F;
  }
};

class Lncosh : public AbstractActivation{
private:
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
  const double log2_ = std::log(2.);

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
    for (int i = 0; i < G.size(); ++i) {
      G(i) = F(i) * std::tanh(Z(i));
    }
    // G.array() = F.array()*Z.array().tanh();
  }
};

// class Activation : public AbstractActivation {
//   using Ptype = std::unique_ptr<AbstractActivation>;
//
//   Ptype m_;
//
// public:
//   explicit Activation(const json &pars) {
//     if (pars["Name"] == "Identity") {
//       m_ = Ptype(new Identity());
//     } else if (pars["Name"] == "Lncosh") {
//       m_ = Ptype(new Lncosh());
//     }
//   }
//
//   virtual inline void activate(const VectorType &Z, VectorType &A) override {
//     return m_->activate(Z, A);
//   }
//
//   virtual inline void apply_jacobian(const VectorType &Z, const VectorType
//   &A,
//                                      const VectorType &F,
//                                      VectorType &G) override {
//     return m_->apply_jacobian(Z, A, F, G);
//   }
// };

} // namespace netket

#endif
