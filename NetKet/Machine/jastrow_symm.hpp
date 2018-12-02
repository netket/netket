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
#include "abstract_machine.hpp"

#ifndef NETKET_JAS_SYMM_HPP
#define NETKET_JAS_SYMM_HPP

namespace netket {

// Jastrow with permutation symmetries
template <typename T>
class JastrowSymm : public AbstractMachine<T> {
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using VectorRefType = typename AbstractMachine<T>::VectorRefType;
  using VectorConstRefType = typename AbstractMachine<T>::VectorConstRefType;
  using VisibleConstType = typename AbstractMachine<T>::VisibleConstType;

  const Hilbert hilbert_;
  Graph graph_;

  std::vector<std::vector<int>> permtable_;
  int permsize_;

  // number of visible units
  int nv_;

  // number of parameters
  int npar_;

  // number of parameters without symmetries
  int nbarepar_;

  // weights
  MatrixType W_;

  // weights with symmetries
  MatrixType Wsymm_;

  VectorType thetas_;
  VectorType thetasnew_;

  Eigen::MatrixXd DerMatSymm_;
  Eigen::MatrixXi Wtemp_;

 public:
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  // constructor
  explicit JastrowSymm(Hilbert hilbert)
      : hilbert_(std::move(hilbert)),
        graph_(hilbert.GetGraph()),
        nv_(hilbert.Size()) {
    Init(graph_);

    SetBareParameters();
  }

  void Init(Graph graph) {
    if (nv_ < 2) {
      throw InvalidInputError(
          "Cannot construct Jastrow states with less than two visible units");
    }

    permtable_ = graph.SymmetryTable();
    permsize_ = permtable_.size();

    for (int i = 0; i < permsize_; i++) {
      assert(int(permtable_[i].size()) == nv_);
    }

    W_.resize(nv_, nv_);
    W_.setZero();
    thetas_.resize(nv_);
    thetasnew_.resize(nv_);

    nbarepar_ = (nv_ * (nv_ - 1)) / 2;

    // Constructing the matrix that maps the bare derivatives to the symmetric
    // ones

    Wtemp_ = Eigen::MatrixXi::Zero(nv_, nv_);

    std::map<int, int> params;
    int k = 0;

    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        for (int l = 0; l < permsize_; l++) {
          int isymm = permtable_[l][i];
          int jsymm = permtable_[l][j];

          if (isymm < 0 || isymm >= nv_ || jsymm < 0 || jsymm >= nv_) {
            std::cerr << "Error in JastrowSymm" << std::endl;
            std::abort();
          }
          Wtemp_(isymm, jsymm) = k;
          Wtemp_(jsymm, isymm) = k;
        }  // l
        k++;
      }  // j
    }    // i

    int nk_unique = 0;

    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        k = Wtemp_(i, j);
        if (params.count(k) == 0) {
          nk_unique++;
          params.insert(std::pair<int, int>(k, nk_unique));
        }
      }
    }

    npar_ = params.size();

    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        if (params.count(Wtemp_(i, j))) {
          Wtemp_(i, j) = params.find(Wtemp_(i, j))->second;
        } else {
          std::cerr << "Error in JastrowSymm" << std::endl;
          std::abort();
        }
        Wtemp_(j, i) = Wtemp_(i, j);
      }
    }

    DerMatSymm_ = Eigen::MatrixXd::Zero(npar_, nbarepar_);
    Wsymm_.resize(npar_, 1);  // used to stay close to RbmSpinSymm class

    int kbare = 0;
    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        int ksymm = Wtemp_(i, j);
        if (ksymm < 1 || ksymm - 1 >= npar_) {
          std::cerr << "Error in JastrowSymm" << std::endl;
          std::abort();
        }
        DerMatSymm_(ksymm - 1, kbare) = 1;
        kbare++;
      }
    }

    InfoMessage() << "Jastrow WF Initizialized with nvisible = " << nv_
                  << std::endl;
    InfoMessage() << "Symmetries are being used : " << npar_
                  << " parameters left, instead of " << nbarepar_ << std::endl;
  }

  int Nvisible() const override { return nv_; }

  // int Nhidden() const { return nh_; }

  int Npar() const override { return npar_; }

  void InitRandomPars(int seed, double sigma) override {
    VectorType par(npar_);

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(par);
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

  // same as RBM
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

  VectorType BareDerLog(VisibleConstType v) {
    VectorType der(nbarepar_);

    int k = 0;
    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        der(k) = v(i) * v(j);
        k++;
      }
    }

    return der;
  }

  // now unchanged w.r.t. RBM spin symm
  VectorType DerLog(VisibleConstType v) override {
    return DerMatSymm_ * BareDerLog(v);
  }

  VectorType GetParameters() override {
    VectorType pars(npar_);

    int k = 0;

    for (int i = 0; i < npar_; i++) {
      pars(k) = Wsymm_(i, 0);
      k++;
    }
    return pars;
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    for (int i = 0; i < npar_; i++) {
      Wsymm_(i, 0) = pars(k);
      k++;
    }

    SetBareParameters();
  }

  void SetBareParameters() {
    for (int i = 0; i < nv_; i++) {
      for (int j = i + 1; j < nv_; j++) {
        W_(i, j) = Wsymm_(Wtemp_(i, j) - 1, 0);
        W_(j, i) = W_(i, j);  // create the lover triangle
        W_(i, i) = T(0);
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

  Hilbert GetHilbert() const override { return hilbert_; }

  void to_json(json &j) const override {
    j["Name"] = "JastrowSymm";
    j["Nvisible"] = nv_;
    j["Wsymm"] = Wsymm_;
  }

  void from_json(const json &pars) override {
    if (pars.at("Name") != "JastrowSymm") {
      throw InvalidInputError(
          "Error while constructing JastrowSymm from Json input");
    }

    if (FieldExists(pars, "Nvisible")) {
      nv_ = pars["Nvisible"];
    }
    if (nv_ != hilbert_.Size()) {
      throw InvalidInputError(
          "Number of visible units is incompatible with given "
          "Hilbert space");
    }

    Init(graph_);

    if (FieldExists(pars, "Wsymm")) {
      Wsymm_ = pars["Wsymm"];
    }

    SetBareParameters();
  }
};

}  // namespace netket

#endif
