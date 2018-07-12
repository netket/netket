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

#ifndef NETKET_URBM_HPP
#define NETKET_URBM_HPP

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
 *
 */
template <typename T>
class URbm : public AbstractMachine<T> {
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;

  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of parameters
  int npar_;

  // weights
  std::vector<MatrixType> W_;

  // visible units bias
  VectorType a_;

  // hidden units bias
  VectorType b_;

  MatrixType thetas_;
  VectorType lnthetas_;
  MatrixType thetasnew_;
  VectorType lnthetasnew_;

  bool usea_;
  bool useb_;

  int mynode_;

  const Hilbert &hilbert_;

 public:
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  // constructor
  explicit URbm(const Hilbert &hilbert, const json &pars)
      : nv_(hilbert.Size()), hilbert_(hilbert) {
    from_json(pars);
  }

  void Init() {
    W_.resize(nh_);
    for (int i = 0; i < nh_; ++i) {
      W_[i].resize(nv_, nv_);
    }
    a_.resize(nv_);
    b_.resize(nh_);

    thetas_.resize(nv_, nh_);
    lnthetas_.resize(nh_);
    thetasnew_.resize(nv_, nh_);
    lnthetasnew_.resize(nh_);

    npar_ = nv_ * nv_ * nh_;

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

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (mynode_ == 0) {
      std::cout << "# URBM Initizialized with nvisible = " << nv_
                << " and nhidden = " << nh_ << std::endl;
      std::cout << "# Using visible bias = " << usea_ << std::endl;
      std::cout << "# Using hidden bias  = " << useb_ << std::endl;
      std::cout << "# Total Number of Parameters = " << npar_ << std::endl;
    }
  }

  int Nvisible() const override { return nv_; }

  int Nhidden() const { return nh_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int seed, double sigma) override {
    VectorType par(npar_);

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(par);
  }

  void InitLookup(const Eigen::VectorXd &v, LookupType &lt) override {
    if (lt.MatrixSize() == 0) {
      lt.AddMatrix(nv_, nh_);
    }

    for (int k = 0; k < nh_; ++k) {
      lt.M(0).col(k) = (W_[k] * v);
    }
  }

  void UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    if (tochange.size() != 0) {
      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        for (int k = 0; k < nh_; ++k) {
          lt.M(0).col(k) += W_[k].col(sf) * (newconf[s] - v(sf));
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

    for (int kk = 0; kk < nh_; ++kk) {
      lnthetas_(kk) = std::tanh((v.transpose() * W_[kk] * v)(0) + b_(kk));
    }

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        der(k) = lnthetas_(p);
        k++;
      }
    }

    for (int kk = 0; kk < nh_; ++kk) {
      for (int j = 0; j < nv_; ++j) {
        for (int i = 0; i < nv_; ++i) {
          der(k) = lnthetas_(kk) * v(i) * v(j);
          ++k;
        }
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

    for (int kk = 0; kk < nh_; kk++) {
      for (int j = 0; j < nv_; j++) {
        for (int i = 0; i < nv_; i++) {
          pars(k) = W_[kk](i, j);
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

    for (int kk = 0; kk < nh_; kk++) {
      for (int j = 0; j < nv_; j++) {
        for (int i = 0; i < nv_; i++) {
          W_[kk](i, j) = pars(k);
          k++;
        }
      }
    }
  }

  // Value of the logarithm of the wave-function
  T LogVal(const Eigen::VectorXd &v) override {
    for (int k = 0; k < nh_; ++k) {
      lnthetas_(k) = URbm::lncosh((v.transpose() * W_[k] * v)(0) + b_(k));
    }
    return (v.dot(a_) + lnthetas_.sum());
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  T LogVal(const Eigen::VectorXd &v, const LookupType &lt) override {
    URbm::lncosh(lt.M(0).transpose() * v + b_, lnthetas_);

    return (v.dot(a_) + lnthetas_.sum());
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      const Eigen::VectorXd &v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    const std::size_t nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);
    VectorType vprime = v;

    for (int k = 0; k < nh_; ++k) {
      thetas_.col(k) = (W_[k] * v);
    }
    URbm::lncosh(v.transpose() * thetas_ + b_, lnthetas_);

    T logtsum = lnthetas_.sum();

    for (std::size_t k = 0; k < nconn; k++) {
      if (tochange[k].size() != 0) {
        thetasnew_ = thetas_;

        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];
          vprime(sf) = newconf[k][s];

          logvaldiffs(k) += a_(sf) * (newconf[k][s] - v(sf));

          for (int kk = 0; kk < nh_; ++kk) {
            thetasnew_.col(kk) += W_[kk].col(sf) * (newconf[k][s] - v(sf));
          }
        }

        URbm::lncosh(thetasnew_.transpose() * vprime + b_, lnthetasnew_);
        logvaldiffs(k) += lnthetasnew_.sum() - logtsum;

        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];
          vprime(sf) = v(sf);
        }
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
    Eigen::VectorXd vprime = v;

    if (tochange.size() != 0) {
      URbm::lncosh(lt.M(0).transpose() * vprime + b_, lnthetas_);

      thetasnew_ = lt.M(0);

      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        vprime(sf) = newconf[s];

        logvaldiff += a_(sf) * (newconf[s] - v(sf));

        for (int k = 0; k < nh_; ++k) {
          thetasnew_.col(k) += W_[k].col(sf) * (newconf[s] - v(sf));
        }
      }
      URbm::lncosh(vprime.transpose() * thetasnew_ + b_, lnthetasnew_);
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

    std::complex<double> res = URbm::lncosh(xr);
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
    j["Machine"]["Name"] = "URbm";
    j["Machine"]["Nvisible"] = nv_;
    j["Machine"]["Nhidden"] = nh_;
    j["Machine"]["UseVisibleBias"] = usea_;
    j["Machine"]["UseHiddenBias"] = useb_;
    j["Machine"]["a"] = a_;
    j["Machine"]["b"] = b_;
    // j["Machine"]["W"] = W_;
  }

  void from_json(const json &pars) override {
    if (pars.at("Machine").at("Name") != "URbm") {
      throw InvalidInputError("Error while constructing URbm from Json input");
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

    usea_ = FieldOrDefaultVal(pars["Machine"], "UseVisibleBias", true);
    useb_ = FieldOrDefaultVal(pars["Machine"], "UseHiddenBias", true);

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
  }
};

}  // namespace netket

#endif
