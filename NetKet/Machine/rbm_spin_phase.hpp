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

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"

#ifndef NETKET_RBM_SPIN_PHASE_HPP
#define NETKET_RBM_SPIN_PHASE_HPP

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
This version has real-valued weights and two RBMs parameterizing phase and
amplitude
 *
 */
class RbmSpinPhase : public AbstractMachine {
  const AbstractHilbert &hilbert_;

  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of parameters
  int npar_;

  // weights
  RealMatrixType W1_;

  // visible units bias
  RealVectorType a1_;

  // hidden units bias
  RealVectorType b1_;

  // weights
  RealMatrixType W2_;

  // visible units bias
  RealVectorType a2_;

  // hidden units bias
  RealVectorType b2_;

  RealVectorType thetas1_;
  RealVectorType thetas2_;
  RealVectorType lnthetas1_;
  RealVectorType lnthetas2_;
  RealVectorType thetasnew1_;
  RealVectorType lnthetasnew1_;
  RealVectorType thetasnew2_;
  RealVectorType lnthetasnew2_;

  bool usea_;
  bool useb_;

  const Complex I_;

 public:
  explicit RbmSpinPhase(const AbstractHilbert &hilbert, int nhidden = 0,
                        int alpha = 0, bool usea = true, bool useb = true)
      : hilbert_(hilbert),
        nv_(hilbert.Size()),
        usea_(usea),
        useb_(useb),
        I_(0, 1) {
    nh_ = std::max(nhidden, alpha * nv_);

    Init();
  }

  void Init() {
    W1_.resize(nv_, nh_);
    a1_.resize(nv_);
    b1_.resize(nh_);

    W2_.resize(nv_, nh_);
    a2_.resize(nv_);
    b2_.resize(nh_);

    thetas1_.resize(nh_);
    thetas2_.resize(nh_);
    lnthetas1_.resize(nh_);
    lnthetas2_.resize(nh_);
    thetasnew1_.resize(nh_);
    lnthetasnew1_.resize(nh_);
    thetasnew2_.resize(nh_);
    lnthetasnew2_.resize(nh_);

    npar_ = nv_ * nh_;

    if (usea_) {
      npar_ += nv_;
    } else {
      a1_.setZero();
      a2_.setZero();
    }

    if (useb_) {
      npar_ += nh_;
    } else {
      b1_.setZero();
      b2_.setZero();
    }

    npar_ *= 2;

    InfoMessage() << "Phase RBM Initizialized with nvisible = " << nv_
                  << " and nhidden = " << nh_ << std::endl;
    InfoMessage() << "Using visible bias = " << usea_ << std::endl;
    InfoMessage() << "Using hidden bias  = " << useb_ << std::endl;
  }

  int Nvisible() const override { return nv_; }

  int Nhidden() const { return nh_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int seed, double sigma) override {
    RealVectorType par(npar_);

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(VectorType(par));
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    if (lt.VectorSize() == 0) {
      lt.AddVector(b1_.size());
      lt.AddVector(b2_.size());
    }
    if (lt.V(0).size() != b1_.size()) {
      lt.V(0).resize(b1_.size());
    }
    if (lt.V(1).size() != b2_.size()) {
      lt.V(1).resize(b2_.size());
    }

    lt.V(0) = (W1_.transpose() * v + b1_);
    lt.V(1) = (W2_.transpose() * v + b2_);
  }

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    if (tochange.size() != 0) {
      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        lt.V(0) += W1_.row(sf) * (newconf[s] - v(sf));
        lt.V(1) += W2_.row(sf) * (newconf[s] - v(sf));
      }
    }
  }

  VectorType DerLog(VisibleConstType v) override {
    LookupType ltnew;
    InitLookup(v, ltnew);
    return DerLog(v, ltnew);
  }

  VectorType DerLog(VisibleConstType v, const LookupType &lt) override {
    VectorType der(npar_);

    const int impar = npar_ / 2;

    if (usea_) {
      der.head(nv_) = v;
      der.segment(impar, nv_) = I_ * v;
    }

    RbmSpin::tanh(lt.V(0).real(), lnthetas1_);
    RbmSpin::tanh(lt.V(1).real(), lnthetas2_);

    if (useb_) {
      der.segment(usea_ * nv_, nh_) = lnthetas1_;
      der.segment(impar + usea_ * nv_, nh_) = I_ * lnthetas2_;
    }

    const int initw = nv_ * usea_ + nh_ * useb_;

    MatrixType wder = (v * lnthetas1_.transpose());
    der.segment(initw, nv_ * nh_) =
        Eigen::Map<VectorType>(wder.data(), nv_ * nh_);

    wder = (v * lnthetas2_.transpose());
    der.segment(impar + initw, nv_ * nh_) =
        I_ * Eigen::Map<VectorType>(wder.data(), nv_ * nh_);

    return der;
  }

  VectorType GetParameters() override {
    VectorType pars(npar_);

    const int impar = npar_ / 2;

    if (usea_) {
      pars.head(nv_) = a1_;
      pars.segment(impar, nv_) = a2_;
    }

    if (useb_) {
      pars.segment(nv_ * usea_, nh_) = b1_;
      pars.segment(impar + nv_ * usea_, nh_) = b2_;
    }

    const int initw = nv_ * usea_ + nh_ * useb_;

    pars.segment(initw, nv_ * nh_) =
        Eigen::Map<RealVectorType>(W1_.data(), nv_ * nh_);
    pars.segment(impar + initw, nv_ * nh_) =
        Eigen::Map<RealVectorType>(W2_.data(), nv_ * nh_);

    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    const int impar = npar_ / 2;

    if (usea_) {
      a1_ = pars.head(nv_).real();
      a2_ = pars.segment(impar, nv_).real();
    }

    if (useb_) {
      b1_ = pars.segment(usea_ * nv_, nh_).real();
      b2_ = pars.segment(impar + usea_ * nv_, nh_).real();
    }

    const int initw = nv_ * usea_ + nh_ * useb_;

    VectorType Wpars = pars.segment(initw, nv_ * nh_);
    W1_ = Eigen::Map<MatrixType>(Wpars.data(), nv_, nh_).real();

    Wpars = pars.segment(impar + initw, nv_ * nh_);
    W2_ = Eigen::Map<MatrixType>(Wpars.data(), nv_, nh_).real();
  }

  // Value of the logarithm of the wave-function
  Complex LogVal(VisibleConstType v) override {
    RbmSpin::lncosh(W1_.transpose() * v + b1_, lnthetas1_);
    RbmSpin::lncosh(W2_.transpose() * v + b2_, lnthetas2_);

    return (v.dot(a1_) + lnthetas1_.sum() +
            I_ * (v.dot(a2_) + lnthetas2_.sum()));
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  Complex LogVal(VisibleConstType v, const LookupType &lt) override {
    RbmSpin::lncosh(lt.V(0).real(), lnthetas1_);
    RbmSpin::lncosh(lt.V(1).real(), lnthetas2_);

    return (v.dot(a1_) + lnthetas1_.sum() +
            I_ * (v.dot(a2_) + lnthetas2_.sum()));
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    const std::size_t nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    thetas1_ = (W1_.transpose() * v + b1_);
    thetas2_ = (W2_.transpose() * v + b2_);

    RbmSpin::lncosh(thetas1_, lnthetas1_);
    RbmSpin::lncosh(thetas2_, lnthetas2_);

    Complex logtsum = lnthetas1_.sum() + I_ * (lnthetas2_.sum());

    for (std::size_t k = 0; k < nconn; k++) {
      if (tochange[k].size() != 0) {
        thetasnew1_ = thetas1_;
        thetasnew2_ = thetas2_;

        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];

          logvaldiffs(k) += a1_(sf) * (newconf[k][s] - v(sf));
          logvaldiffs(k) += I_ * a2_(sf) * (newconf[k][s] - v(sf));

          thetasnew1_ += W1_.row(sf) * (newconf[k][s] - v(sf));
          thetasnew2_ += W2_.row(sf) * (newconf[k][s] - v(sf));
        }

        RbmSpin::lncosh(thetasnew1_, lnthetasnew1_);
        RbmSpin::lncosh(thetasnew2_, lnthetasnew2_);
        logvaldiffs(k) +=
            lnthetasnew1_.sum() + I_ * lnthetasnew2_.sum() - logtsum;
      }
    }
    return logvaldiffs;
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped Version using pre-computed look-up tables for efficiency
  // on a small number of spin flips
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override {
    Complex logvaldiff = 0.;

    if (tochange.size() != 0) {
      RbmSpin::lncosh(lt.V(0).real(), lnthetas1_);
      RbmSpin::lncosh(lt.V(1).real(), lnthetas2_);

      thetasnew1_ = lt.V(0).real();
      thetasnew2_ = lt.V(1).real();

      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];

        logvaldiff += a1_(sf) * (newconf[s] - v(sf));
        logvaldiff += I_ * a2_(sf) * (newconf[s] - v(sf));

        thetasnew1_ += W1_.row(sf) * (newconf[s] - v(sf));
        thetasnew2_ += W2_.row(sf) * (newconf[s] - v(sf));
      }

      RbmSpin::lncosh(thetasnew1_, lnthetasnew1_);
      RbmSpin::lncosh(thetasnew2_, lnthetasnew2_);
      logvaldiff += (lnthetasnew1_.sum() - lnthetas1_.sum());
      logvaldiff += I_ * (lnthetasnew2_.sum() - lnthetas2_.sum());
    }
    return logvaldiff;
  }

  inline static double lncosh(double x) {
    const double xp = std::abs(x);
    if (xp <= 12.) {
      return std::log(std::cosh(xp));
    } else {
      const static double log2v = std::log(2.);
      return xp - log2v;
    }
  }

  const AbstractHilbert &GetHilbert() const noexcept override {
    return hilbert_;
  }

  void to_json(json &j) const override {
    j["Name"] = "RbmSpinPhase";
    j["Nvisible"] = nv_;
    j["Nhidden"] = nh_;
    j["UseVisibleBias"] = usea_;
    j["UseHiddenBias"] = useb_;
    j["a1"] = a1_;
    j["b1"] = b1_;
    j["W1"] = W1_;
    j["a2"] = a2_;
    j["b2"] = b2_;
    j["W2"] = W2_;
  }

  void from_json(const json &pars) override {
    std::string name = FieldVal<std::string>(pars, "Name");
    if (name != "RbmSpinPhase") {
      throw InvalidInputError(
          "Error while constructing RbmSpinPhase from input parameters");
    }

    if (FieldExists(pars, "Nvisible")) {
      nv_ = FieldVal<int>(pars, "Nvisible");
    }
    if (nv_ != hilbert_.Size()) {
      throw InvalidInputError(
          "Number of visible units is incompatible with given "
          "Hilbert space");
    }

    if (FieldExists(pars, "Nhidden")) {
      nh_ = FieldVal<int>(pars, "Nhidden");
    } else {
      nh_ = nv_ * double(FieldVal<double>(pars, "Alpha"));
    }

    usea_ = FieldOrDefaultVal(pars, "UseVisibleBias", true);
    useb_ = FieldOrDefaultVal(pars, "UseHiddenBias", true);

    Init();

    // Loading parameters, if defined in the input
    if (FieldExists(pars, "a1")) {
      a1_ = FieldVal<RealVectorType>(pars, "a1");
      a2_ = FieldVal<RealVectorType>(pars, "a2");
    } else {
      a1_.setZero();
      a2_.setZero();
    }

    if (FieldExists(pars, "b1")) {
      b1_ = FieldVal<RealVectorType>(pars, "b1");
      b2_ = FieldVal<RealVectorType>(pars, "b2");
    } else {
      b1_.setZero();
      b2_.setZero();
    }
    if (FieldExists(pars, "W1")) {
      W1_ = FieldVal<RealMatrixType>(pars, "W1");
      W2_ = FieldVal<RealMatrixType>(pars, "W2");
    }
  }

  bool IsHolomorphic() override { return false; }
};

}  // namespace netket

#endif
