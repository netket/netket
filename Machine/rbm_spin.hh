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

#include "Lookup/lookup.hh"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>

#ifndef NETKET_RBM_SPIN_HH
#define NETKET_RBM_SPIN_HH

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
 *
 */
template <typename T> class RbmSpin : public AbstractMachine<T> {

  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;

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

  VectorType thetas_;
  VectorType lnthetas_;
  VectorType thetasnew_;
  VectorType lnthetasnew_;

  bool usea_;
  bool useb_;

  int mynode_;

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
    W_.resize(nv_, nh_);
    a_.resize(nv_);
    b_.resize(nh_);

    thetas_.resize(nh_);
    lnthetas_.resize(nh_);
    thetasnew_.resize(nh_);
    lnthetasnew_.resize(nh_);

    npar_ = nv_ * nh_;

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
      std::cout << "# RBM Initizialized with nvisible = " << nv_
                << " and nhidden = " << nh_ << std::endl;
      std::cout << "# Using visible bias = " << usea_ << std::endl;
      std::cout << "# Using hidden bias  = " << useb_ << std::endl;
    }
  }

  int Nvisible() const { return nv_; }

  int Nhidden() const { return nh_; }

  int Npar() const { return npar_; }

  void InitRandomPars(int seed, double sigma) {

    VectorType par(npar_);

    RandomGaussian(par, seed, sigma);

    SetParameters(par);
  }

  void InitLookup(const Eigen::VectorXd &v, LookupType &lt) {
    if (lt.VectorSize() == 0) {
      lt.AddVector(b_.size());
    }
    if (lt.V(0).size() != b_.size()) {
      lt.V(0).resize(b_.size());
    }

    lt.V(0) = (W_.transpose() * v + b_);
  }

  void UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf, LookupType &lt) {

    if (tochange.size() != 0) {

      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        lt.V(0) += W_.row(sf) * (newconf[s] - v(sf));
      }
    }
  }

  VectorType DerLog(const Eigen::VectorXd &v) {
    VectorType der(npar_);

    int k = 0;

    if (usea_) {
      for (; k < nv_; k++) {
        der(k) = v(k);
      }
    }

    RbmSpin::tanh(W_.transpose() * v + b_, lnthetas_);

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

  VectorType GetParameters() {

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

    for (int i = 0; i < nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        pars(k) = W_(i, j);
        k++;
      }
    }

    return pars;
  }

  void SetParameters(const VectorType &pars) {
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

    for (int i = 0; i < nv_; i++) {
      for (int j = 0; j < nh_; j++) {
        W_(i, j) = pars(k);
        k++;
      }
    }
  }

  // Value of the logarithm of the wave-function
  T LogVal(const Eigen::VectorXd &v) {
    RbmSpin::lncosh(W_.transpose() * v + b_, lnthetas_);

    return (v.dot(a_) + lnthetas_.sum());
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  T LogVal(const Eigen::VectorXd &v, LookupType &lt) {
    RbmSpin::lncosh(lt.V(0), lnthetas_);

    return (v.dot(a_) + lnthetas_.sum());
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(const Eigen::VectorXd &v,
                        const std::vector<std::vector<int>> &tochange,
                        const std::vector<std::vector<double>> &newconf) {

    const std::size_t nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    thetas_ = (W_.transpose() * v + b_);
    RbmSpin::lncosh(thetas_, lnthetas_);

    T logtsum = lnthetas_.sum();

    for (std::size_t k = 0; k < nconn; k++) {

      if (tochange[k].size() != 0) {

        thetasnew_ = thetas_;

        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];

          logvaldiffs(k) += a_(sf) * (newconf[k][s] - v(sf));

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
               const std::vector<double> &newconf, const LookupType &lt) {

    T logvaldiff = 0.;

    if (tochange.size() != 0) {

      RbmSpin::lncosh(lt.V(0), lnthetas_);

      thetasnew_ = lt.V(0);

      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];

        logvaldiff += a_(sf) * (newconf[s] - v(sf));

        thetasnew_ += W_.row(sf) * (newconf[s] - v(sf));
      }

      RbmSpin::lncosh(thetasnew_, lnthetasnew_);
      logvaldiff += (lnthetasnew_.sum() - lnthetas_.sum());
    }
    return logvaldiff;
  }

  static void RandomGaussian(Eigen::Matrix<double, Eigen::Dynamic, 1> &par,
                             int seed, double sigma) {
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0, sigma);
    for (int i = 0; i < par.size(); i++) {
      par(i) = distribution(generator);
    }
  }

  static void
  RandomGaussian(Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> &par,
                 int seed, double sigma) {
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0, sigma);
    for (int i = 0; i < par.size(); i++) {
      par(i) = std::complex<double>(distribution(generator),
                                    distribution(generator));
    }
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

  void to_json(json &j) const {
    j["Machine"]["Name"] = "RbmSpin";
    j["Machine"]["Nvisible"] = nv_;
    j["Machine"]["Nhidden"] = nh_;
    j["Machine"]["UseVisibleBias"] = usea_;
    j["Machine"]["UseHiddenBias"] = useb_;
    j["Machine"]["a"] = a_;
    j["Machine"]["b"] = b_;
    j["Machine"]["W"] = W_;
  }

  void from_json(const json &pars) {

    if (pars.at("Machine").at("Name") != "RbmSpin") {
      if (mynode_ == 0) {
        std::cerr << "# Error while constructing RbmSpin from Json input"
                  << std::endl;
      }
      std::abort();
    }

    if (FieldExists(pars["Machine"], "Nvisible")) {
      nv_ = pars["Machine"]["Nvisible"];
    }
    if (nv_ != hilbert_.Size()) {
      if (mynode_ == 0) {
        std::cerr << "# Number of visible units is incompatible with given "
                     "Hilbert space"
                  << std::endl;
      }
      std::abort();
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
    if (FieldExists(pars["Machine"], "W")) {
      W_ = pars["Machine"]["W"];
    }
  }
};

} // namespace netket

#endif
