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
#include "Lookup/lookup.hpp"
#include "Utils/all_utils.hpp"

#ifndef NETKET_RBM_SPIN_HPP
#define NETKET_RBM_SPIN_HPP

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
 *
 */
template <typename T>
class RbmSpin : public AbstractMachine<T> {
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;

  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of wavelets
  int nw_;

  // number of parameters
  int npar_;

  // weights
  MatrixType W_;

  // predefined wavelet and wavelet weights
  MatrixType Wavelets_;
  VectorType W_wavelets_;

  // visible units bias
  VectorType a_;

  // hidden units bias
  VectorType b_;

  VectorType thetas_;
  VectorType lnthetas_;
  VectorType thetasnew_;
  VectorType lnthetasnew_;

  bool usea_;
  bool useb_;

  // flag to know wether wavelets are used or not
  bool usewavelets_;

  const Hilbert &hilbert_;

 public:
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  // constructor
  explicit RbmSpin(const Hilbert &hilbert, const json &pars)
      : nv_(hilbert.Size()), hilbert_(hilbert) {
    from_json(pars);
  }

  void Init() {
    if (usewavelets_) {
      W_wavelets_.resize(nw_)

      Wavelets_.resize(nv_, nw_)
      // initialize wavelets using gabor wavelets
      for (int i = 0; i < nv_; i++) {
        for (int j = 0; j < nw_; j++) {
            // NOTE: there I need to know if I'm in 1D or 2D
            Wavelets_(i, j) = 0.;
            k++;
        }
      }
    } else {
      W_.resize(nv_, nh_);
    }

    a_.resize(nv_);
    b_.resize(nh_);

    thetas_.resize(nh_);
    lnthetas_.resize(nh_);
    thetasnew_.resize(nh_);
    lnthetasnew_.resize(nh_);

    if (usewavelets_) {
      npar_ = nw_;
    } else {
      npar_ = nv_ * nh_;
    }

    if (usea_) {
      npar_ += nv_;
    } else {
      a_.setZero();
    }

    if (useb_) {
      npar_ += nh_;
    } else {
      b_.setZero();
    }

    if (usewavelets_) {
      InfoMessage() << "RBM Initizialized with nvisible = " << nv_
                    << " using wavelets with nwavelets = " << nw_ << std::endl;
    } else {
      InfoMessage() << "RBM Initizialized with nvisible = " << nv_
                    << " and nhidden = " << nh_ << std::endl;
    }
    InfoMessage() << "Using visible bias = " << usea_ << std::endl;
    InfoMessage() << "Using hidden bias  = " << useb_ << std::endl;
  }

  int Nvisible() const override { return nv_; }

  int Nhidden() const { return nh_; }

  int Nwavelets() const { return nw_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int seed, double sigma) override {
    VectorType par(npar_);

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(par);
  }

  void InitLookup(const Eigen::VectorXd &v, LookupType &lt) override {
    if (lt.VectorSize() == 0) {
      lt.AddVector(b_.size());
    }
    if (lt.V(0).size() != b_.size()) {
      lt.V(0).resize(b_.size());
    }

    if (usewavelets_) {
      // NOTE: I don't know if the Vector matrix mul is corrrect and if I understood correctly the convolution
      lt.V(0) = (W_wavelets_ * Wavelets_.transpose() * v + b_);
    } else {
      lt.V(0) = (W_.transpose() * v + b_);
    }
  }

  void UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    if (tochange.size() != 0) {
      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        if (usewavelets_) {
          lt.V(0) += (Wavelets_ * W_wavelets_).row(sf) * (newconf[s] - v(sf));
        } else {
          lt.V(0) += W_.row(sf) * (newconf[s] - v(sf));
        }
      }
    }
  }

  VectorType DerLog(const Eigen::VectorXd &v) override {
    VectorType der(npar_);

    int k = 0;

    if (usea_) {
      for (; k < nv_; k++) {
        der(k) = v(k);
      }
    }

    if (usewavelets_) {
      // I'm not sure either
      RbmSpin::tanh(W_wavelets_ * Wavelets_.transpose() * v + b_, lnthetas_);
    } else {
      RbmSpin::tanh(W_.transpose() * v + b_, lnthetas_);
    }

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        der(k) = lnthetas_(p);
        k++;
      }
    }

    for (int i = 0; i < nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        der(k) = lnthetas_(j) * v(i);
        k++;
      }
    }
    return der;
  }

  VectorType GetParameters() override {
    VectorType pars(npar_);

    int k = 0;

    if (usea_) {
      for (; k < nv_; k++) {
        pars(k) = a_(k);
      }
    }

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        pars(k) = b_(p);
        k++;
      }
    }

    if (usewavelets_) {
      for (int i = 0; i < nw_; i++) {
        pars(k) = W_wavelets_(i);
        k++;
      }
    } else {
      for (int i = 0; i < nv_; i++) {
        for (int j = 0; j < nh_; j++) {
          pars(k) = W_(i, j);
          k++;
        }
      }
    }

    return pars;
  }

  void SetParameters(const VectorType &pars) override {
    int k = 0;

    if (usea_) {
      for (; k < nv_; k++) {
        a_(k) = pars(k);
      }
    }

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        b_(p) = pars(k);
        k++;
      }
    }

    if (usewavelets_) {
      for (int i = 0; i < nw_; i++) {
        W_wavelets_(i) = pars(k);
        k++;
      }
    } else {
      for (int i = 0; i < nv_; i++) {
        for (int j = 0; j < nh_; j++) {
          W_(i, j) = pars(k);
          k++;
        }
      }
    }
  }

  // Value of the logarithm of the wave-function
  T LogVal(const Eigen::VectorXd &v) override {
    if (usewavelets_) {
      // I'm not sure either
      RbmSpin::lncosh(W_wavelets_ * Wavelets_.transpose() * v + b_, lnthetas_);
    } else {
      RbmSpin::lncosh(W_.transpose() * v + b_, lnthetas_);
    }

    return (v.dot(a_) + lnthetas_.sum());
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  T LogVal(const Eigen::VectorXd &v, const LookupType &lt) override {
    RbmSpin::lncosh(lt.V(0), lnthetas_);

    return (v.dot(a_) + lnthetas_.sum());
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      const Eigen::VectorXd &v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    const std::size_t nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    if (usewavelets_) {
      // I'm not sure either
      thetas_ = (W_wavelets_ * Wavelets_.transpose() * v + b_);
    } else {
      thetas_ = (W_.transpose() * v + b_);
    }
    RbmSpin::lncosh(thetas_, lnthetas_);

    T logtsum = lnthetas_.sum();

    for (std::size_t k = 0; k < nconn; k++) {
      if (tochange[k].size() != 0) {
        thetasnew_ = thetas_;

        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];

          logvaldiffs(k) += a_(sf) * (newconf[k][s] - v(sf));

        if (usewavelets_) {
          // I'm not sure either
          thetasnew_ += (Wavelets_ * W_wavelets_).row(sf) * (newconf[k][s] - v(sf));
        } else {
          thetasnew_ += W_.row(sf) * (newconf[k][s] - v(sf));
        }

        RbmSpin::lncosh(thetasnew_, lnthetasnew_);
        logvaldiffs(k) += lnthetasnew_.sum() - logtsum;
      }
    }
    return logvaldiffs;
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped Version using pre-computed look-up tables for efficiency
  // on a small number of spin flips
  T LogValDiff(const Eigen::VectorXd &v, const std::vector<int> &tochange,
               const std::vector<double> &newconf,
               const LookupType &lt) override {
    T logvaldiff = 0.;

    if (tochange.size() != 0) {
      RbmSpin::lncosh(lt.V(0), lnthetas_);

      thetasnew_ = lt.V(0);

      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];

        logvaldiff += a_(sf) * (newconf[s] - v(sf));

        if (usewavelets_) {
          // I'm not sure either
          thetasnew_ += (Wavelets_ * W_wavelets_).row(sf) * (newconf[s] - v(sf));
        } else {
          thetasnew_ += W_.row(sf) * (newconf[s] - v(sf));
        }
      }

      RbmSpin::lncosh(thetasnew_, lnthetasnew_);
      logvaldiff += (lnthetasnew_.sum() - lnthetas_.sum());
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

  // ln(cos(x)) for std::complex argument
  // the modulus is computed by means of the previously defined function
  // for real argument
  inline static std::complex<double> lncosh(std::complex<double> x) {
    const double xr = x.real();
    const double xi = x.imag();

    std::complex<double> res = RbmSpin::lncosh(xr);
    res += std::log(
        std::complex<double>(std::cos(xi), std::tanh(xr) * std::sin(xi)));

    return res;
  }

  static void tanh(const VectorType &x, VectorType &y) {
    assert(y.size() >= x.size());
    y = Eigen::tanh(x.array());
  }

  static void lncosh(const VectorType &x, VectorType &y) {
    assert(y.size() >= x.size());
    for (int i = 0; i < x.size(); i++) {
      y(i) = lncosh(x(i));
    }
  }

  const Hilbert &GetHilbert() const { return hilbert_; }

  void to_json(json &j) const override {
    j["Machine"]["Name"] = "RbmSpin";
    j["Machine"]["Nvisible"] = nv_;
    j["Machine"]["Nhidden"] = nh_;
    j["Machine"]["Nwavelets"] = nw_;
    j["Machine"]["UseVisibleBias"] = usea_;
    j["Machine"]["UseHiddenBias"] = useb_;
    j["Machine"]["UseWavelets"] = usewavelets_;
    j["Machine"]["a"] = a_;
    j["Machine"]["b"] = b_;
    j["Machine"]["W"] = W_;
    j["Machine"]["W_wavelets"] = W_wavelets_;
    // Wavelets do not need to be stored since they are predefined
  }

  void from_json(const json &pars) override {
    if (pars.at("Machine").at("Name") != "RbmSpin") {
      throw InvalidInputError(
          "Error while constructing RbmSpin from Json input");
    }

    if (FieldExists(pars["Machine"], "Nvisible")) {
      nv_ = pars["Machine"]["Nvisible"];
    }
    if (nv_ != hilbert_.Size()) {
      throw InvalidInputError(
          "Number of visible units is incompatible with given "
          "Hilbert space");
    }

    if (FieldExists(pars["Machine"], "Nhidden")) {
      nh_ = FieldVal(pars["Machine"], "Nhidden");
    } else {
      nh_ = nv_ * double(FieldVal(pars["Machine"], "Alpha"));
    }

    if (FieldExists(pars["Machine"], "Nwavelets")) {
      nw_ = pars["Machine"]["Nvisible"];
    }

    usea_ = FieldOrDefaultVal(pars["Machine"], "UseVisibleBias", true);
    useb_ = FieldOrDefaultVal(pars["Machine"], "UseHiddenBias", true);
    usewavelets_ = FieldOrDefaultVal(pars["Machine"], "UseWavelets", false);

    Init();

    // Loading parameters, if defined in the input
    if (FieldExists(pars["Machine"], "a")) {
      a_ = pars["Machine"]["a"];
    } else {
      a_.setZero();
    }

    if (FieldExists(pars["Machine"], "b")) {
      b_ = pars["Machine"]["b"];
    } else {
      b_.setZero();
    }
    if (FieldExists(pars["Machine"], "W")) {
      W_ = pars["Machine"]["W"];
    }
    if (FieldExists(pars["Machine"], "W_wavelets")) {
      W_wavelets_ = pars["Machine"]["W_wavelets"];
    }
  }
};

}  // namespace netket

#endif
