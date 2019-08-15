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

#ifndef NETKET_RBM_SPIN_HPP
#define NETKET_RBM_SPIN_HPP

#include <cmath>
#include <unordered_map>
#include "Machine/abstract_machine.hpp"
#include "Utils/array_hasher.hpp"

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
 *
 */
class RbmSpin : public AbstractMachine {
  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of parameters
  int npar_;

  // weights
  MatrixType W_;

  // visible units bias
  VectorType a_;

  // hidden units bias
  VectorType b_;

  VectorType lnthetas_;

  bool usea_;
  bool useb_;

  bool cache_vals_;

  std::unordered_map<VisibleType, Complex, EigenArrayHasher<VisibleType>,
                     EigenArrayEqualityComparison<VisibleType>>
      log_vals_cache_;

 public:
  RbmSpin(std::shared_ptr<const AbstractHilbert> hilbert, int nhidden,
          int alpha, bool usea, bool useb, bool cache_vals);

  int Nvisible() const override;
  int Npar() const override;
  /*constexpr*/ int Nhidden() const noexcept { return nh_; }

  VectorType DerLogSingle(VisibleConstType v, const any &lt) override;
  Complex LogValSingle(VisibleConstType v, const any &lt) override;

  VectorType GetParameters() override;
  void SetParameters(VectorConstRefType pars) override;

  void Save(const std::string &filename) const override;
  void Load(const std::string &filename) override;

  bool IsHolomorphic() const noexcept override;

  Complex LogValImpl(VisibleConstType v);

  static double lncosh(double x) {
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
  static Complex lncosh(Complex x) {
    const double xr = x.real();
    const double xi = x.imag();

    Complex res = RbmSpin::lncosh(xr);
    res += std::log(Complex(std::cos(xi), std::tanh(xr) * std::sin(xi)));

    return res;
  }

  static void tanh(VectorConstRefType x, VectorType &y) {
    assert(y.size() >= x.size());
    y = x.array().tanh();
  }

  static void tanh(RealVectorConstRefType x, RealVectorType &y) {
    assert(y.size() >= x.size());
    y = x.array().tanh();
  }

  static void lncosh(VectorConstRefType x, VectorType &y) {
    assert(y.size() >= x.size());
    for (int i = 0; i < x.size(); i++) {
      y(i) = lncosh(x(i));
    }
  }

  static void lncosh(RealVectorConstRefType x, RealVectorType &y) {
    assert(y.size() >= x.size());
    for (int i = 0; i < x.size(); i++) {
      y(i) = lncosh(x(i));
    }
  }

 private:
  inline void Init();
  VectorType DerLogSingleImpl(VisibleConstType v, const any &lookup);
};

}  // namespace netket

#endif
