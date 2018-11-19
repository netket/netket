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
//
// by G. Mazzola, May-Aug 2018

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"

#ifndef NETKET_JASTROW_HPP
#define NETKET_JASTROW_HPP

namespace netket {

/** Jastrow machine class.
 *
 */
template <typename T>
class Jastrow : public AbstractMachine<T> {
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using VectorRefType = typename AbstractMachine<T>::VectorRefType;
  using VectorConstRefType = typename AbstractMachine<T>::VectorConstRefType;
  using VisibleConstType = typename AbstractMachine<T>::VisibleConstType;

  std::shared_ptr<const AbstractHilbert> hilbert_;

  // number of visible units
  int nv_;

  // number of parameters
  int npar_;

  // weights
  MatrixType W_;

  // buffers
  VectorType thetas_;
  VectorType thetasnew_;

 public:
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  // constructor
  explicit Jastrow(std::shared_ptr<const AbstractHilbert> hilbert)
      : hilbert_(hilbert), nv_(hilbert->Size()) {
    Init();
  }

  void Init() {
    W_.resize(nv_, nv_);

    npar_ = (nv_ * (nv_ - 1)) / 2;

    thetas_.resize(nv_);
    thetasnew_.resize(nv_);

    InfoMessage() << "Jastrow WF Initizialized with nvisible = " << nv_
                  << " and nparams = " << npar_ << std::endl;
  }

  int Nvisible() const override { return nv_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int seed, double sigma) override {
    VectorType par(npar_);

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(par);
  }

  VectorType GetParameters() override {
    VectorType pars(npar_);

    int k = 0;

    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        pars(k) = W_(i, j);
        k++;
      }
    }

    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        W_(i, j) = pars(k);
        W_(j, i) = W_(i, j);  // create the lower triangle
        W_(i, i) = T(0);
        k++;
      }
    }
  }

  void InitLookup(VisibleConstType v, LookupType &lt) override {
    if (lt.VectorSize() == 0) {
      lt.AddVector(v.size());
    }
    if (lt.V(0).size() != v.size()) {
      lt.V(0).resize(v.size());
    }

    lt.V(0) = (W_.transpose() * v);  // does not matter the transpose W is symm
  }

  // same as for the RBM
  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    if (tochange.size() != 0) {
      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        lt.V(0) += W_.row(sf) * (newconf[s] - v(sf));
      }
    }
  }

  T LogVal(VisibleConstType v) override { return 0.5 * v.dot(W_ * v); }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  T LogVal(VisibleConstType v, const LookupType &lt) override {
    return 0.5 * v.dot(lt.V(0));
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    const std::size_t nconn = tochange.size();
    VectorType logvaldiffs = VectorType::Zero(nconn);

    thetas_ = (W_.transpose() * v);
    T logtsum = 0.5 * v.dot(thetas_);

    for (std::size_t k = 0; k < nconn; k++) {
      if (tochange[k].size() != 0) {
        thetasnew_ = thetas_;
        Eigen::VectorXd vnew(v);

        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];

          thetasnew_ += W_.row(sf) * (newconf[k][s] - v(sf));
          vnew(sf) = newconf[k][s];
        }

        logvaldiffs(k) = 0.5 * vnew.dot(thetasnew_) - logtsum;
      }
    }
    return logvaldiffs;
  }

  T LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
               const std::vector<double> &newconf,
               const LookupType &lt) override {
    T logvaldiff = 0.;

    if (tochange.size() != 0) {
      T logtsum = 0.5 * v.dot(lt.V(0));
      thetasnew_ = lt.V(0);
      Eigen::VectorXd vnew(v);

      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];

        thetasnew_ += W_.row(sf) * (newconf[s] - v(sf));
        vnew(sf) = newconf[s];
      }

      logvaldiff = 0.5 * vnew.dot(thetasnew_) - logtsum;
    }

    return logvaldiff;
  }

  VectorType DerLog(VisibleConstType v) override {
    VectorType der(npar_);

    int k = 0;

    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        der(k) = v(i) * v(j);
        k++;
      }
    }

    return der;
  }

  std::shared_ptr<const AbstractHilbert> GetHilbert() const override {
    return hilbert_;
  }

  void to_json(json &j) const override {
    j["Machine"]["Name"] = "Jastrow";
    j["Machine"]["Nvisible"] = nv_;
    j["Machine"]["W"] = W_;
  }

  void from_json(const json &pars) override {
    if (pars.at("Name") != "Jastrow") {
      throw InvalidInputError(
          "Error while constructing Jastrow from Json input");
    }

    if (FieldExists(pars, "Nvisible")) {
      nv_ = pars["Nvisible"];
    }
    if (nv_ != hilbert_->Size()) {
      throw InvalidInputError(
          "Number of visible units is incompatible with given "
          "Hilbert space");
    }

    Init();

    if (FieldExists(pars, "W")) {
      W_ = pars["W"];
    }
  }
};
}  // namespace netket

#endif
