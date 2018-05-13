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
#include "abstract_machine.hh"
#include "rbm_spin.hh"
#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#ifndef NETKET_RBM_MULTIVAL_HH
#define NETKET_RBM_MULTIVAL_HH

namespace netket {

// Restricted Boltzman Machine wave function
// for generic (finite) local hilbert space
template <typename T> class RbmMultival : public AbstractMachine<T> {

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

  Eigen::VectorXd localconfs_;
  Eigen::MatrixXd mask_;

  Eigen::VectorXd vtilde_;

  // local size of hilbert space
  int ls_;

  std::map<double, int> confindex_;

public:
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  // Json constructor
  explicit RbmMultival(const Hilbert &hilbert, const json &pars)
      : nv_(hilbert.Size()), hilbert_(hilbert), ls_(hilbert.LocalSize()) {

    from_json(pars);
  }

  void Init() {

    W_.resize(nv_ * ls_, nh_);
    a_.resize(nv_ * ls_);
    b_.resize(nh_);

    thetas_.resize(nh_);
    lnthetas_.resize(nh_);
    thetasnew_.resize(nh_);
    lnthetasnew_.resize(nh_);

    npar_ = nv_ * nh_ * ls_;

    if (usea_) {
      npar_ += nv_ * ls_;
    } else {
      a_.setZero();
    }

    if (useb_) {
      npar_ += nh_;
    } else {
      b_.setZero();
    }

    auto localstates = hilbert_.LocalStates();

    localconfs_.resize(nv_ * ls_);
    for (int i = 0; i < nv_ * ls_; i += ls_) {
      for (int j = 0; j < ls_; j++) {
        localconfs_(i + j) = localstates[j];
      }
    }

    mask_.resize(nv_ * ls_, nv_);
    mask_.setZero();

    for (int i = 0; i < nv_ * ls_; i++) {
      mask_(i, i / ls_) = 1;
    }

    for (int i = 0; i < ls_; i++) {
      confindex_[localstates[i]] = i;
    }

    vtilde_.resize(nv_ * ls_);

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (mynode_ == 0) {
      std::cout << "# RBM Multival Initizialized with nvisible = " << nv_
                << " and nhidden = " << nh_ << std::endl;
      std::cout << "# Using visible bias = " << usea_ << std::endl;
      std::cout << "# Using hidden bias  = " << useb_ << std::endl;
      std::cout << "# Local size is      = " << ls_ << std::endl;
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
    ComputeTheta(v, lt.V(0));
  }

  void UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf, LookupType &lt) {

    if (tochange.size() != 0) {

      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        const int oldtilde = confindex_[v[sf]];
        const int newtilde = confindex_[newconf[s]];

        lt.V(0) -= W_.row(ls_ * sf + oldtilde);
        lt.V(0) += W_.row(ls_ * sf + newtilde);
      }
    }
  }

  VectorType DerLog(const Eigen::VectorXd &v) {
    VectorType der(npar_);
    der.setZero();

    int k = 0;

    ComputeTheta(v, thetas_);

    if (usea_) {
      for (; k < nv_ * ls_; k++) {
        der(k) = vtilde_(k);
      }
    }

    RbmSpin<T>::tanh(thetas_, lnthetas_);

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        der(k) = lnthetas_(p);
        k++;
      }
    }

    for (int i = 0; i < nv_ * ls_; i++) {
      for (int j = 0; j < nh_; j++) {
        der(k) = lnthetas_(j) * vtilde_(i);
        k++;
      }
    }
    return der;
  }

  VectorType GetParameters() {

    VectorType pars(npar_);

    int k = 0;

    if (usea_) {
      for (; k < nv_ * ls_; k++) {
        pars(k) = a_(k);
      }
    }

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        pars(k) = b_(p);
        k++;
      }
    }

    for (int i = 0; i < nv_ * ls_; i++) {
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
      for (; k < nv_ * ls_; k++) {
        a_(k) = pars(k);
      }
    }

    if (useb_) {
      for (int p = 0; p < nh_; p++) {
        b_(p) = pars(k);
        k++;
      }
    }

    for (int i = 0; i < nv_ * ls_; i++) {
      for (int j = 0; j < nh_; j++) {
        W_(i, j) = pars(k);
        k++;
      }
    }
  }

  // Value of the logarithm of the wave-function
  T LogVal(const Eigen::VectorXd &v) {
    ComputeTheta(v, thetas_);
    RbmSpin<T>::lncosh(thetas_, lnthetas_);

    return (vtilde_.dot(a_) + lnthetas_.sum());
  }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  T LogVal(const Eigen::VectorXd &v, LookupType &lt) {
    RbmSpin<T>::lncosh(lt.V(0), lnthetas_);

    ComputeVtilde(v, vtilde_);
    return (vtilde_.dot(a_) + lnthetas_.sum());
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being changed
  VectorType LogValDiff(const Eigen::VectorXd &v,
                        const std::vector<std::vector<int>> &tochange,
                        const std::vector<std::vector<double>> &newconf) {

    const std::size_t nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    ComputeTheta(v, thetas_);
    RbmSpin<T>::lncosh(thetas_, lnthetas_);

    T logtsum = lnthetas_.sum();

    for (std::size_t k = 0; k < nconn; k++) {

      if (tochange[k].size() != 0) {

        thetasnew_ = thetas_;

        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];
          const int oldtilde = confindex_[v[sf]];
          const int newtilde = confindex_[newconf[k][s]];

          logvaldiffs(k) -= a_(ls_ * sf + oldtilde);
          logvaldiffs(k) += a_(ls_ * sf + newtilde);

          thetasnew_ -= W_.row(ls_ * sf + oldtilde);
          thetasnew_ += W_.row(ls_ * sf + newtilde);
        }

        RbmSpin<T>::lncosh(thetasnew_, lnthetasnew_);
        logvaldiffs(k) += lnthetasnew_.sum() - logtsum;
      }
    }
    return logvaldiffs;
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being changed Version using pre-computed look-up tables for efficiency
  // on a small number of local changes
  T LogValDiff(const Eigen::VectorXd &v, const std::vector<int> &tochange,
               const std::vector<double> &newconf, const LookupType &lt) {

    T logvaldiff = 0.;

    if (tochange.size() != 0) {

      RbmSpin<T>::lncosh(lt.V(0), lnthetas_);

      thetasnew_ = lt.V(0);

      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        const int oldtilde = confindex_[v[sf]];
        const int newtilde = confindex_[newconf[s]];

        logvaldiff -= a_(ls_ * sf + oldtilde);
        logvaldiff += a_(ls_ * sf + newtilde);

        thetasnew_ -= W_.row(ls_ * sf + oldtilde);
        thetasnew_ += W_.row(ls_ * sf + newtilde);
      }

      RbmSpin<T>::lncosh(thetasnew_, lnthetasnew_);
      logvaldiff += (lnthetasnew_.sum() - lnthetas_.sum());
    }
    return logvaldiff;
  }

  // Computhes the values of the theta pseudo-angles
  inline void ComputeTheta(const Eigen::VectorXd &v, VectorType &theta) {
    ComputeVtilde(v, vtilde_);
    theta = (W_.transpose() * vtilde_ + b_);
  }

  inline void ComputeVtilde(const Eigen::VectorXd &v, Eigen::VectorXd &vtilde) {
    auto t = (localconfs_.array() == (mask_ * v).array());
    vtilde = t.template cast<double>();
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

  const Hilbert &GetHilbert() const { return hilbert_; }

  void to_json(json &j) const {
    j["Machine"]["Name"] = "RbmMultival";
    j["Machine"]["Nvisible"] = nv_;
    j["Machine"]["Nhidden"] = nh_;
    j["Machine"]["LocalSize"] = ls_;
    j["Machine"]["UseVisibleBias"] = usea_;
    j["Machine"]["UseHiddenBias"] = useb_;
    j["Machine"]["a"] = a_;
    j["Machine"]["b"] = b_;
    j["Machine"]["W"] = W_;
  }

  void from_json(const json &pars) {

    if (pars.at("Machine").at("Name") != "RbmMultival") {
      if (mynode_ == 0) {
        std::cerr << "# Error while constructing RbmMultival from Json input"
                  << std::endl;
      }
      std::abort();
    }

    if (FieldExists(pars["Machine"], "Nvisible")) {
      nv_ = pars["Machine"]["Nvisible"];
    }
    if (nv_ != hilbert_.Size()) {
      if (mynode_ == 0) {
        std::cerr << "# Loaded wave-function has incompatible Hilbert space"
                  << std::endl;
      }
      std::abort();
    }

    if (FieldExists(pars["Machine"], "LocalSize")) {
      ls_ = pars["Machine"]["LocalSize"];
    }
    if (ls_ != hilbert_.LocalSize()) {
      if (mynode_ == 0) {
        std::cerr << "# Loaded wave-function has incompatible Hilbert space"
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
