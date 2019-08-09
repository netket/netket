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

#include "rbm_spin_symm.hpp"

#include "Machine/rbm_spin.hpp"
#include "Utils/json_utils.hpp"
#include "Utils/log_cosh.hpp"
#include "Utils/messages.hpp"

namespace netket {

RbmSpinSymm::RbmSpinSymm(std::shared_ptr<const AbstractHilbert> hilbert,
                         int alpha, bool usea, bool useb)
    : AbstractMachine(hilbert),
      graph_(hilbert->GetGraph()),
      nv_(hilbert->Size()),
      alpha_(alpha),
      usea_(usea),
      useb_(useb) {
  Init(graph_);
  SetBareParameters();
}

void RbmSpinSymm::RbmSpinSymm::Init(const AbstractGraph &graph) {
  permtable_ = graph.SymmetryTable();
  permsize_ = permtable_.size();
  nh_ = (alpha_ * permsize_);

  for (int i = 0; i < permsize_; i++) {
    assert(int(permtable_[i].size()) == nv_);
  }

  W_.resize(nv_, nh_);
  a_.resize(nv_);
  b_.resize(nh_);

  thetas_.resize(nh_);
  thetasnew_.resize(nh_);

  Wsymm_.resize(nv_, alpha_);
  bsymm_.resize(alpha_);

  npar_ = nv_ * alpha_;
  nbarepar_ = nv_ * nh_;

  if (usea_) {
    npar_ += 1;
    nbarepar_ += nv_;
  } else {
    asymm_ = 0;
    a_.setZero();
  }

  if (useb_) {
    npar_ += alpha_;
    nbarepar_ += nh_;
  } else {
    bsymm_.setZero();
    b_.setZero();
  }

  // Constructing the matrix that maps the bare derivatives to the symmetric
  // ones
  DerMatSymm_ = Eigen::MatrixXd::Zero(npar_, nbarepar_);

  int k = 0;
  int kbare = 0;

  if (usea_) {
    // derivative with respect to a
    for (int p = 0; p < nv_; p++) {
      DerMatSymm_(k, p) = 1;
      kbare++;
    }
    k++;
  }

  if (useb_) {
    // derivatives with respect to b
    for (int p = 0; p < nh_; p++) {
      int ksymm = std::floor(double(p) / double(permsize_));
      DerMatSymm_(ksymm + k, kbare) = 1;
      kbare++;
    }
    k += alpha_;
  }

  // derivatives with respect to W
  for (int i = 0; i < nv_; i++) {
    for (int j = 0; j < nh_; j++) {
      int isymm = permtable_.at(j % permsize_).at(i);
      int jsymm = std::floor(double(j) / double(permsize_));
      int ksymm = jsymm + alpha_ * isymm;

      DerMatSymm_(ksymm + k, kbare) = 1;

      kbare++;
    }
  }

  InfoMessage() << "RBM Initizialized with nvisible = " << nv_
                << " and nhidden = " << nh_ << std::endl;
  InfoMessage() << "Symmetries are being used : " << npar_
                << " parameters left, instead of " << nbarepar_ << std::endl;
}

int RbmSpinSymm::Nvisible() const { return nv_; }

int RbmSpinSymm::Npar() const { return npar_; }

RbmSpinSymm::VectorType RbmSpinSymm::BareDerLog(VisibleConstType v) {
  VectorType der(nbarepar_);

  int k = 0;

  if (usea_) {
    for (; k < nv_; k++) {
      der(k) = v(k);
    }
  }

  RbmSpin::tanh(W_.transpose() * v + b_, thetas_);

  if (useb_) {
    for (int p = 0; p < nh_; p++) {
      der(k) = thetas_(p);
      k++;
    }
  }

  for (int i = 0; i < nv_; i++) {
    for (int j = 0; j < nh_; j++) {
      der(k) = thetas_(j) * v(i);
      k++;
    }
  }
  return der;
}

RbmSpinSymm::VectorType RbmSpinSymm::DerLogSingle(VisibleConstType v,
                                                  const any & /*lt*/) {
  return DerMatSymm_ * BareDerLog(v);
}

RbmSpinSymm::VectorType RbmSpinSymm::GetParameters() {
  VectorType pars(npar_);

  int k = 0;

  if (usea_) {
    pars(k) = asymm_;
    k++;
  }

  if (useb_) {
    for (int p = 0; p < (alpha_); p++) {
      pars(k) = bsymm_(p);
      k++;
    }
  }

  for (int i = 0; i < nv_; i++) {
    for (int j = 0; j < alpha_; j++) {
      pars(k) = Wsymm_(i, j);
      k++;
    }
  }

  return pars;
}

void RbmSpinSymm::SetParameters(VectorConstRefType pars) {
  int k = 0;

  if (usea_) {
    asymm_ = pars(k);
    k++;
  } else {
    asymm_ = 0;
  }

  if (useb_) {
    for (int p = 0; p < alpha_; p++) {
      bsymm_(p) = pars(k);
      k++;
    }
  } else {
    bsymm_ = VectorType::Zero(alpha_);
  }

  for (int i = 0; i < nv_; i++) {
    for (int j = 0; j < alpha_; j++) {
      Wsymm_(i, j) = pars(k);
      k++;
    }
  }

  SetBareParameters();
}

void RbmSpinSymm::SetBareParameters() {
  // Setting the bare values of the RBM parameters
  for (int i = 0; i < nv_; i++) {
    a_(i) = asymm_;
  }

  for (int j = 0; j < nh_; j++) {
    int jsymm = std::floor(double(j) / double(permsize_));
    b_(j) = bsymm_(jsymm);
  }

  for (int i = 0; i < nv_; i++) {
    for (int j = 0; j < nh_; j++) {
      int jsymm = std::floor(double(j) / double(permsize_));
      W_(i, j) = Wsymm_(permtable_[j % permsize_][i], jsymm);
    }
  }
}

// Value of the logarithm of the wave-function
// using pre-computed look-up tables for efficiency
Complex RbmSpinSymm::LogValSingle(VisibleConstType v, const any & /*lt*/) {
  return v.dot(a_) + SumLogCosh(W_.transpose() * v + b_);
}

void RbmSpinSymm::Save(const std::string &filename) const {
  json state;
  state["Name"] = "RbmSpinSymm";
  state["Nvisible"] = nv_;
  state["Alpha"] = alpha_;
  state["UseVisibleBias"] = usea_;
  state["UseHiddenBias"] = useb_;
  state["asymm"] = asymm_;
  state["bsymm"] = bsymm_;
  state["Wsymm"] = Wsymm_;
  WriteJsonToFile(state, filename);
}

void RbmSpinSymm::Load(const std::string &filename) {
  auto const pars = ReadJsonFromFile(filename);
  if (pars.at("Name") != "RbmSpinSymm") {
    throw InvalidInputError(
        "Error while constructing RbmSpinSymm from Json input");
  }

  if (FieldExists(pars, "Nvisible")) {
    nv_ = pars["Nvisible"];
  }
  if (nv_ != GetHilbert().Size()) {
    throw InvalidInputError(
        "Number of visible units is incompatible with given "
        "Hilbert space");
  }

  alpha_ = FieldVal(pars, "Alpha", "Machine");

  usea_ = FieldOrDefaultVal(pars, "UseVisibleBias", true);
  useb_ = FieldOrDefaultVal(pars, "UseHiddenBias", true);

  Init(graph_);

  // Loading parameters, if defined in the input
  if (FieldExists(pars, "asymm")) {
    asymm_ = pars["asymm"].get<Complex>();
  } else {
    asymm_ = 0;
  }

  if (FieldExists(pars, "bsymm")) {
    bsymm_ = pars["bsymm"];
  } else {
    bsymm_.setZero();
  }
  if (FieldExists(pars, "Wsymm")) {
    Wsymm_ = pars["Wsymm"];
  }

  SetBareParameters();
}

bool RbmSpinSymm::IsHolomorphic() const noexcept { return true; }

}  // namespace netket
