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

#ifndef NETKET_NDM_SPIN_PHASE_HPP
#define NETKET_NDM_SPIN_PHASE_HPP

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Machine/rbm_spin.hpp"
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"
#include "abstract_density_matrix.hpp"

namespace netket {

/** Neural Density Matrix machine class with spin 1/2 hidden units.
This version has real-valued weights and two NDMs parameterizing phase and
amplitude
 *
 */
class NdmSpinPhase : public AbstractDensityMatrix {
  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of ancillary units
  int na_;

  // number of parameters
  int npar_;

  // visible units bias
  RealVectorType b1_;
  RealVectorType b2_;

  // hidden units bias
  RealVectorType h1_;
  RealVectorType h2_;

  // ancillary units bias
  RealVectorType d1_;

  // hidden unit weights
  RealMatrixType W1_;
  RealMatrixType W2_;

  // ancillary unit weights
  RealMatrixType U1_;
  RealMatrixType U2_;

  // Caches
  RealVectorType thetas_r1_;
  RealVectorType thetas_r2_;
  RealVectorType thetas_c1_;
  RealVectorType thetas_c2_;
  RealVectorType lnthetas_r1_;
  RealVectorType lnthetas_r2_;
  RealVectorType lnthetas_c1_;
  RealVectorType lnthetas_c2_;
  RealVectorType thetasnew_r1_;
  RealVectorType thetasnew_r2_;
  RealVectorType thetasnew_c1_;
  RealVectorType thetasnew_c2_;
  RealVectorType lnthetasnew_r1_;
  RealVectorType lnthetasnew_r2_;
  RealVectorType lnthetasnew_c1_;
  RealVectorType lnthetasnew_c2_;

  RealVectorType thetas_a1_;
  RealVectorType thetas_a2_;
  RealVectorType thetasnew_a1_;
  RealVectorType thetasnew_a2_;
  VectorType pi_;
  VectorType lnpi_;
  VectorType lnpinew_;

  bool useb_;
  bool useh_;
  bool used_;

  const Complex I_;

 public:
  explicit NdmSpinPhase(std::shared_ptr<const AbstractHilbert> hilbert,
                        int nhidden = 0, int nancilla = 0, int alpha = 0,
                        int beta = 0, bool useb = true, bool useh = true,
                        bool used = true)
      : AbstractDensityMatrix(hilbert),
        nv_(hilbert->Size()),
        useb_(useb),
        useh_(useh),
        used_(used),
        I_(0, 1) {
    nh_ = std::max(nhidden, alpha * nv_);
    na_ = std::max(nancilla, beta * nv_);
    Init();
  }

  void Init() {
    W1_.resize(nv_, nh_);
    U1_.resize(nv_, na_);
    b1_.resize(nv_);
    h1_.resize(nh_);
    d1_.resize(na_);

    W2_.resize(nv_, nh_);
    U2_.resize(nv_, na_);
    b2_.resize(nv_);
    h2_.resize(nh_);

    thetas_r1_.resize(nh_);
    thetas_r2_.resize(nh_);
    thetas_c1_.resize(nh_);
    thetas_c2_.resize(nh_);

    lnthetas_r1_.resize(nh_);
    lnthetas_r2_.resize(nh_);
    lnthetas_c1_.resize(nh_);
    lnthetas_c2_.resize(nh_);

    thetasnew_r1_.resize(nh_);
    thetasnew_r2_.resize(nh_);
    thetasnew_c1_.resize(nh_);
    thetasnew_c2_.resize(nh_);

    lnthetasnew_r1_.resize(nh_);
    lnthetasnew_r2_.resize(nh_);
    lnthetasnew_c1_.resize(nh_);
    lnthetasnew_c2_.resize(nh_);

    thetas_a1_.resize(na_);
    thetas_a2_.resize(na_);
    thetasnew_a1_.resize(na_);
    thetasnew_a2_.resize(na_);
    pi_.resize(na_);
    lnpi_.resize(na_);
    lnpinew_.resize(na_);

    npar_ = 2 * nv_ * (nh_ + na_);

    if (useb_) {
      npar_ += 2 * nv_;
    } else {
      b1_.setZero();
      b2_.setZero();
    }

    if (useh_) {
      npar_ += 2 * nh_;
    } else {
      h1_.setZero();
      h2_.setZero();
    }

    if (used_) {
      npar_ += na_;
    } else {
      d1_.setZero();
    }

    InfoMessage() << "Phase NDM Initizialized with nvisible = " << nv_
                  << " and nhidden  = " << nh_ << " and nancilla = " << na_
                  << std::endl;
    InfoMessage() << "Using visible   bias = " << useb_ << std::endl;
    InfoMessage() << "Using hidden    bias  = " << useh_ << std::endl;
    InfoMessage() << "Using ancillary bias  = " << used_ << std::endl;
  }

  int Nvisible() const override { return nv_; }

  int Nhidden() const { return nh_; }

  int Nancilla() const { return na_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int seed, double sigma) override {
    RealVectorType par(npar_);

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(VectorType(par));
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    if (lt.VectorSize() == 0) {
      lt.AddVector(h1_.size());  // row 1
      lt.AddVector(h2_.size());  // row 2
      lt.AddVector(h1_.size());  // col 1
      lt.AddVector(h2_.size());  // col 2
      lt.AddVector(d1_.size());  // ancilla modulus
      lt.AddVector(d1_.size());  // ancilla phase
    }
    if (lt.V(0).size() != h1_.size()) {
      lt.V(0).resize(h1_.size());
    }
    if (lt.V(1).size() != h2_.size()) {
      lt.V(1).resize(h2_.size());
    }
    if (lt.V(2).size() != h1_.size()) {
      lt.V(2).resize(h1_.size());
    }
    if (lt.V(3).size() != h2_.size()) {
      lt.V(3).resize(h2_.size());
    }
    if (lt.V(4).size() != d1_.size()) {
      lt.V(4).resize(d1_.size());
    }
    if (lt.V(5).size() != d1_.size()) {
      lt.V(5).resize(d1_.size());
    }

    VisibleConstType vr = v.head(GetHilbertPhysical().Size());
    VisibleConstType vc = v.tail(GetHilbertPhysical().Size());

    lt.V(0) = (W1_.transpose() * vr + h1_);
    lt.V(1) = (W2_.transpose() * vr + h2_);
    lt.V(2) = (W1_.transpose() * vc + h1_);
    lt.V(3) = (W2_.transpose() * vc + h2_);

    lt.V(4) = (0.5 * U1_.transpose() * (vr + vc) + d1_);
    lt.V(5) = (0.5 * U2_.transpose() * (vr - vc));
  }

  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    VisibleConstType vr = v.head(GetHilbertPhysical().Size());
    VisibleConstType vc = v.tail(GetHilbertPhysical().Size());

    if (tochange.size() != 0) {
      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        if (sf < Nvisible()) {
          lt.V(0) += W1_.row(sf) * (newconf[s] - vr(sf));
          lt.V(1) += W2_.row(sf) * (newconf[s] - vr(sf));

          lt.V(4) += 0.5 * U1_.row(sf) * (newconf[s] - vr(sf));
          lt.V(5) += 0.5 * U2_.row(sf) * (newconf[s] - vr(sf));
        } else {
          const int sfc = sf - Nvisible();
          lt.V(2) += W1_.row(sfc) * (newconf[s] - vc(sfc));
          lt.V(3) += W2_.row(sfc) * (newconf[s] - vc(sfc));

          lt.V(4) += 0.5 * U1_.row(sfc) * (newconf[s] - vc(sfc));
          lt.V(5) -= 0.5 * U2_.row(sfc) * (newconf[s] - vc(sfc));
        }
      }
    }
  }

  VectorType DerLog(VisibleConstType v) override {
    LookupType ltnew;
    InitLookup(v, ltnew);
    return DerLog(v, ltnew);
  }

  VectorType DerLog(VisibleConstType v, const LookupType &lt) override {
    VisibleConstType vr = v.head(GetHilbertPhysical().Size());
    VisibleConstType vc = v.tail(GetHilbertPhysical().Size());

    VectorType der(npar_);

    const int impar = (npar_ + na_ * used_) / 2;

    if (useb_) {
      der.head(nv_) = 0.5 * (vr + vc);
      der.segment(impar, nv_) = I_ * 0.5 * (vr - vc);
    }

    RbmSpin::tanh(lt.V(0).real(), lnthetas_r1_);
    RbmSpin::tanh(lt.V(1).real(), lnthetas_r2_);
    RbmSpin::tanh(lt.V(2).real(), lnthetas_c1_);
    RbmSpin::tanh(lt.V(3).real(), lnthetas_c2_);

    if (useh_) {
      der.segment(useb_ * nv_, nh_) = 0.5 * (lnthetas_r1_ + lnthetas_c1_);
      der.segment(impar + useb_ * nv_, nh_) =
          I_ * 0.5 * (lnthetas_r2_ - lnthetas_c2_);
    }

    thetas_a1_ = 0.5 * U1_.transpose() * (vr + vc) + d1_;
    thetas_a2_ = 0.5 * U2_.transpose() * (vr - vc);
    RbmSpin::tanh(lt.V(4).real() + I_ * lt.V(5).real(), lnpi_);

    if (used_) {
      der.segment(useb_ * nv_ + useh_ * nh_, na_) = lnpi_;
    }

    const int initw_1 = nv_ * useb_ + nh_ * useh_ + na_ * used_;
    const int initw_2 = nv_ * useb_ + nh_ * useh_;

    MatrixType wder =
        0.5 * (vr * lnthetas_r1_.transpose() + vc * lnthetas_c1_.transpose());
    der.segment(initw_1, nv_ * nh_) =
        Eigen::Map<VectorType>(wder.data(), nv_ * nh_);

    wder = 0.5 * I_ *
           (vr * lnthetas_r2_.transpose() - vc * lnthetas_c2_.transpose());
    der.segment(impar + initw_2, nv_ * nh_) =
        Eigen::Map<VectorType>(wder.data(), nv_ * nh_);

    const int initu_1 = initw_1 + nv_ * nh_;
    const int initu_2 = initw_2 + nv_ * nh_;

    MatrixType uder = 0.5 * (vr + vc) * lnpi_.transpose();
    der.segment(initu_1, nv_ * na_) =
        Eigen::Map<VectorType>(uder.data(), nv_ * na_);

    uder = 0.5 * I_ * (vr - vc) * lnpi_.transpose();
    der.segment(impar + initu_2, nv_ * na_) =
        Eigen::Map<VectorType>(uder.data(), nv_ * na_);

    return der;
  }

  VectorType GetParameters() override {
    VectorType pars(npar_);

    const int impar = (npar_ + na_ * used_) / 2;

    if (useb_) {
      pars.head(nv_) = b1_;
      pars.segment(impar, nv_) = b2_;
    }

    if (useh_) {
      pars.segment(nv_ * useb_, nh_) = h1_;
      pars.segment(impar + nv_ * useb_, nh_) = h2_;
    }

    if (used_) {
      pars.segment(useb_ * nv_ + useh_ * nh_, na_) = d1_;
    }

    const int initw_1 = nv_ * useb_ + nh_ * useh_ + na_ * used_;
    const int initw_2 = nv_ * useb_ + nh_ * useh_;

    pars.segment(initw_1, nv_ * nh_) =
        Eigen::Map<RealVectorType>(W1_.data(), nv_ * nh_);
    pars.segment(impar + initw_2, nv_ * nh_) =
        Eigen::Map<RealVectorType>(W2_.data(), nv_ * nh_);

    const int initu_1 = initw_1 + nv_ * nh_;
    const int initu_2 = initw_2 + nv_ * nh_;

    pars.segment(initu_1, nv_ * na_) =
        Eigen::Map<RealVectorType>(U1_.data(), nv_ * na_);
    pars.segment(impar + initu_2, nv_ * na_) =
        Eigen::Map<RealVectorType>(U2_.data(), nv_ * na_);

    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    const int impar = (npar_ + na_ * used_) / 2;

    if (useb_) {
      b1_ = pars.head(nv_).real();
      b2_ = pars.segment(impar, nv_).real();
    }

    if (useh_) {
      h1_ = pars.segment(useb_ * nv_, nh_).real();
      h2_ = pars.segment(impar + useb_ * nv_, nh_).real();
    }

    if (used_) {
      d1_ = pars.segment(useb_ * nv_ + useh_ * nh_, na_).real();
    }

    const int initw_1 = nv_ * useb_ + nh_ * useh_ + na_ * used_;
    const int initw_2 = nv_ * useb_ + nh_ * useh_;

    VectorType Wpars = pars.segment(initw_1, nv_ * nh_);
    W1_ = Eigen::Map<MatrixType>(Wpars.data(), nv_, nh_).real();

    Wpars = pars.segment(impar + initw_2, nv_ * nh_);
    W2_ = Eigen::Map<MatrixType>(Wpars.data(), nv_, nh_).real();

    const int initu_1 = initw_1 + nv_ * nh_;
    const int initu_2 = initw_2 + nv_ * nh_;

    VectorType Upars = pars.segment(initu_1, nv_ * na_);
    U1_ = Eigen::Map<MatrixType>(Upars.data(), nv_, na_).real();

    Upars = pars.segment(impar + initu_2, nv_ * na_);
    U2_ = Eigen::Map<MatrixType>(Upars.data(), nv_, na_).real();
  }

  // Value of the logarithm of the wave-function
  Complex LogVal(VisibleConstType v) override {
    VisibleConstType vr = v.head(GetHilbertPhysical().Size());
    VisibleConstType vc = v.tail(GetHilbertPhysical().Size());

    RbmSpin::lncosh(W1_.transpose() * vr + h1_, lnthetas_r1_);
    RbmSpin::lncosh(W2_.transpose() * vr + h2_, lnthetas_r2_);
    RbmSpin::lncosh(W1_.transpose() * vc + h1_, lnthetas_c1_);
    RbmSpin::lncosh(W2_.transpose() * vc + h2_, lnthetas_c2_);

    thetas_a1_ = 0.5 * U1_.transpose() * (vr + vc) + d1_;
    thetas_a2_ = 0.5 * U2_.transpose() * (vr - vc);
    RbmSpin::lncosh(thetas_a1_ + I_ * thetas_a2_, lnpi_);

    auto gamma_1 =
        0.5 * (lnthetas_r1_.sum() + lnthetas_c1_.sum() + (vr + vc).dot(b1_));

    auto gamma_2 =
        0.5 * (lnthetas_r2_.sum() - lnthetas_c2_.sum() + (vr - vc).dot(b2_));

    return gamma_1 + I_ * gamma_2 + lnpi_.sum();
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  Complex LogVal(VisibleConstType v, const LookupType &lt) override {
    VisibleConstType vr = v.head(GetHilbertPhysical().Size());
    VisibleConstType vc = v.tail(GetHilbertPhysical().Size());

    RbmSpin::lncosh(lt.V(0).real(), lnthetas_r1_);
    RbmSpin::lncosh(lt.V(1).real(), lnthetas_r2_);
    RbmSpin::lncosh(lt.V(2).real(), lnthetas_c1_);
    RbmSpin::lncosh(lt.V(3).real(), lnthetas_c2_);
    RbmSpin::lncosh(lt.V(4).real() + I_ * lt.V(5).real(), lnpi_);

    auto gamma_1 =
        0.5 * (lnthetas_r1_.sum() + lnthetas_c1_.sum() + (vr + vc).dot(b1_));
    auto gamma_2 =
        0.5 * (lnthetas_r2_.sum() - lnthetas_c2_.sum() + (vr - vc).dot(b2_));

    return gamma_1 + I_ * gamma_2 + lnpi_.sum();
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    VisibleConstType vr = v.head(GetHilbertPhysical().Size());
    VisibleConstType vc = v.tail(GetHilbertPhysical().Size());

    const std::size_t nconn = tochange.size();

    thetas_r1_ = (W1_.transpose() * vr + h1_);
    thetas_r2_ = (W2_.transpose() * vr + h2_);
    thetas_c1_ = (W1_.transpose() * vc + h1_);
    thetas_c2_ = (W2_.transpose() * vc + h2_);

    RbmSpin::lncosh(thetas_r1_, lnthetas_r1_);
    RbmSpin::lncosh(thetas_r2_, lnthetas_r2_);
    RbmSpin::lncosh(thetas_c1_, lnthetas_c1_);
    RbmSpin::lncosh(thetas_c2_, lnthetas_c2_);

    thetas_a1_ = 0.5 * U1_.transpose() * (vr + vc) + d1_;
    thetas_a2_ = 0.5 * U2_.transpose() * (vr - vc);
    RbmSpin::lncosh(thetas_a1_ + I_ * thetas_a2_, lnpi_);

    Complex logtsum = 0.5 * (lnthetas_r1_.sum() + lnthetas_c1_.sum()) +
                      0.5 * I_ * (lnthetas_r2_.sum() - lnthetas_c2_.sum()) +
                      lnpi_.sum();

    VectorType logvaldiffs = VectorType::Zero(nconn);
    for (std::size_t k = 0; k < nconn; k++) {
      if (tochange[k].size() != 0) {
        thetasnew_r1_ = thetas_r1_;
        thetasnew_r2_ = thetas_r2_;
        thetasnew_c1_ = thetas_c1_;
        thetasnew_c2_ = thetas_c2_;

        thetasnew_a1_ = thetas_a1_;
        thetasnew_a2_ = thetas_a2_;

        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];

          if (sf < Nvisible()) {
            logvaldiffs(k) += 0.5 * b1_(sf) * (newconf[k][s] - vr(sf));
            logvaldiffs(k) += 0.5 * I_ * b2_(sf) * (newconf[k][s] - vr(sf));

            thetasnew_r1_ += W1_.row(sf) * (newconf[k][s] - vr(sf));
            thetasnew_r2_ += W2_.row(sf) * (newconf[k][s] - vr(sf));
            thetasnew_a1_ += 0.5 * U1_.row(sf) * (newconf[k][s] - vr(sf));
            thetasnew_a2_ += 0.5 * U2_.row(sf) * (newconf[k][s] - vr(sf));
          } else {
            const int sfc = tochange[k][s] - Nvisible();
            logvaldiffs(k) += 0.5 * b1_(sfc) * (newconf[k][s] - vc(sfc));
            logvaldiffs(k) -= 0.5 * I_ * b2_(sfc) * (newconf[k][s] - vc(sfc));

            thetasnew_c1_ += W1_.row(sfc) * (newconf[k][s] - vc(sfc));
            thetasnew_c2_ += W2_.row(sfc) * (newconf[k][s] - vc(sfc));
            thetasnew_a1_ += 0.5 * U1_.row(sfc) * (newconf[k][s] - vc(sfc));
            thetasnew_a2_ -= 0.5 * U2_.row(sfc) * (newconf[k][s] - vc(sfc));
          }
        }
        RbmSpin::lncosh(thetasnew_r1_, lnthetasnew_r1_);
        RbmSpin::lncosh(thetasnew_r2_, lnthetasnew_r2_);
        RbmSpin::lncosh(thetasnew_c1_, lnthetasnew_c1_);
        RbmSpin::lncosh(thetasnew_c2_, lnthetasnew_c2_);
        RbmSpin::lncosh(thetasnew_a1_ + I_ * thetasnew_a2_, lnpinew_);

        logvaldiffs(k) +=
            0.5 * (lnthetasnew_r1_.sum() + lnthetasnew_c1_.sum()) +
            0.5 * I_ * (lnthetasnew_r2_.sum() - lnthetasnew_c2_.sum()) +
            lnpinew_.sum() - logtsum;
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
    VisibleConstType vr = v.head(GetHilbertPhysical().Size());
    VisibleConstType vc = v.tail(GetHilbertPhysical().Size());

    Complex logvaldiff = 0.;

    if (tochange.size() != 0) {
      RbmSpin::lncosh(lt.V(0).real(), lnthetas_r1_);
      RbmSpin::lncosh(lt.V(1).real(), lnthetas_r2_);
      RbmSpin::lncosh(lt.V(2).real(), lnthetas_c1_);
      RbmSpin::lncosh(lt.V(3).real(), lnthetas_c2_);
      RbmSpin::lncosh(lt.V(4).real() + I_ * lt.V(5).real(), lnpi_);

      thetasnew_r1_ = lt.V(0).real();
      thetasnew_r2_ = lt.V(1).real();
      thetasnew_c1_ = lt.V(2).real();
      thetasnew_c2_ = lt.V(3).real();
      thetasnew_a1_ = lt.V(4).real();
      thetasnew_a2_ = lt.V(5).real();

      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];

        if (sf < Nvisible()) {
          logvaldiff += 0.5 * b1_(sf) * (newconf[s] - vr(sf));
          logvaldiff += 0.5 * I_ * b2_(sf) * (newconf[s] - vr(sf));

          thetasnew_r1_ += W1_.row(sf) * (newconf[s] - vr(sf));
          thetasnew_r2_ += W2_.row(sf) * (newconf[s] - vr(sf));
          thetasnew_a1_ += 0.5 * U1_.row(sf) * (newconf[s] - vr(sf));
          thetasnew_a2_ += 0.5 * U2_.row(sf) * (newconf[s] - vr(sf));
        } else {
          const int sfc = tochange[s] - Nvisible();
          logvaldiff += 0.5 * b1_(sfc) * (newconf[s] - vc(sfc));
          logvaldiff -= 0.5 * I_ * b2_(sfc) * (newconf[s] - vc(sfc));

          thetasnew_c1_ += W1_.row(sfc) * (newconf[s] - vc(sfc));
          thetasnew_c2_ += W2_.row(sfc) * (newconf[s] - vc(sfc));
          thetasnew_a1_ += 0.5 * U1_.row(sfc) * (newconf[s] - vc(sfc));
          thetasnew_a2_ -= 0.5 * U2_.row(sfc) * (newconf[s] - vc(sfc));
        }
      }

      RbmSpin::lncosh(thetasnew_r1_, lnthetasnew_r1_);
      RbmSpin::lncosh(thetasnew_r2_, lnthetasnew_r2_);
      RbmSpin::lncosh(thetasnew_c1_, lnthetasnew_c1_);
      RbmSpin::lncosh(thetasnew_c2_, lnthetasnew_c2_);
      RbmSpin::lncosh(thetasnew_a1_ + I_ * thetasnew_a2_, lnpinew_);

      logvaldiff += 0.5 * (lnthetasnew_r1_.sum() + lnthetasnew_c1_.sum());
      logvaldiff -= 0.5 * (lnthetas_r1_.sum() + lnthetas_c1_.sum());
      logvaldiff += 0.5 * I_ * (lnthetasnew_r2_.sum() - lnthetasnew_c2_.sum());
      logvaldiff -= 0.5 * I_ * (lnthetas_r2_.sum() - lnthetas_c2_.sum());
      logvaldiff += (lnpinew_.sum() - lnpi_.sum());
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

  void to_json(json &j) const override {
    j["Name"] = "NdmSpinPhase";
    j["Nvisible"] = nv_;
    j["Nhidden"] = nh_;
    j["Nancilla"] = na_;
    j["UseVisibleBias"] = useb_;
    j["UseHiddenBias"] = useh_;
    j["UseAncillaBias"] = used_;
    j["b1"] = b1_;
    j["h1"] = h1_;
    j["d1"] = d1_;
    j["W1"] = W1_;
    j["U1"] = U1_;

    j["b2"] = b2_;
    j["h2"] = h2_;
    j["W2"] = W2_;
    j["U2"] = U2_;
  }

  void from_json(const json &pars) override {
    std::string name = FieldVal<std::string>(pars, "Name");
    if (name != "NdmSpinPhase") {
      throw InvalidInputError(
          "Error while constructing RbmSpinPhase from input parameters");
    }

    if (FieldExists(pars, "Nvisible")) {
      nv_ = FieldVal<int>(pars, "Nvisible");
    }
    if (nv_ != GetHilbertPhysical().Size()) {
      throw InvalidInputError(
          "Number of visible units is incompatible with given "
          "Hilbert space");
    }

    if (FieldExists(pars, "Nhidden")) {
      nh_ = FieldVal<int>(pars, "Nhidden");
    } else {
      nh_ = nv_ * double(FieldVal<double>(pars, "Alpha"));
    }

    if (FieldExists(pars, "Nancilla")) {
      na_ = FieldVal<int>(pars, "Nancilla");
    } else {
      na_ = nv_ * double(FieldVal<double>(pars, "Beta"));
    }

    useb_ = FieldOrDefaultVal(pars, "UseVisibleBias", true);
    useh_ = FieldOrDefaultVal(pars, "UseHiddenBias", true);
    used_ = FieldOrDefaultVal(pars, "UseAncillaBias", true);

    Init();

    // Loading parameters, if defined in the input
    if (FieldExists(pars, "b1")) {
      b1_ = FieldVal<RealVectorType>(pars, "b1");
      b2_ = FieldVal<RealVectorType>(pars, "b2");
    } else {
      b1_.setZero();
      b2_.setZero();
    }

    if (FieldExists(pars, "h1")) {
      h1_ = FieldVal<RealVectorType>(pars, "h1");
      h2_ = FieldVal<RealVectorType>(pars, "h2");
    } else {
      h1_.setZero();
      h2_.setZero();
    }

    if (FieldExists(pars, "d1")) {
      d1_ = FieldVal<RealVectorType>(pars, "d1");
    } else {
      d1_.setZero();
    }

    if (FieldExists(pars, "W1")) {
      W1_ = FieldVal<RealMatrixType>(pars, "W1");
      W2_ = FieldVal<RealMatrixType>(pars, "W2");
    }

    if (FieldExists(pars, "U1")) {
      U1_ = FieldVal<RealMatrixType>(pars, "U1");
      U2_ = FieldVal<RealMatrixType>(pars, "U2");
    }
  }

  bool IsHolomorphic() override { return false; }
};

}  // namespace netket

#endif  // NETKET_NDM_SPIN_PHASE_HPP
