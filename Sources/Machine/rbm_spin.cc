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

#include "rbm_spin.hpp"

#include "Utils/json_utils.hpp"
#include "Utils/log_cosh.hpp"
#include "Utils/messages.hpp"

namespace netket {

RbmSpin::RbmSpin(std::shared_ptr<const AbstractHilbert> hilbert, int nhidden,
                 int alpha, bool usea, bool useb)
    : AbstractMachine(hilbert), nv_(hilbert->Size()), usea_(usea), useb_(useb) {
  nh_ = std::max(nhidden, alpha * nv_);
  Init();
}

int RbmSpin::Nvisible() const { return nv_; }

int RbmSpin::Npar() const { return npar_; }

void RbmSpin::Init() {
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

  InfoMessage() << "RBM Initizialized with nvisible = " << nv_
                << " and nhidden = " << nh_ << std::endl;
  InfoMessage() << "Using visible bias = " << usea_ << std::endl;
  InfoMessage() << "Using hidden bias  = " << useb_ << std::endl;
}

RbmSpin::VectorType RbmSpin::DerLogSingle(VisibleConstType v,
                                          const any & /*lookup*/) {
  VectorType der(npar_);

  if (usea_) {
    der.head(nv_) = v;
  }

  RbmSpin::tanh(W_.transpose() * v + b_, lnthetas_);

  if (useb_) {
    der.segment(usea_ * nv_, nh_) = lnthetas_;
  }

  MatrixType wder = (v * lnthetas_.transpose());
  der.tail(nv_ * nh_) = Eigen::Map<VectorType>(wder.data(), nv_ * nh_);

  return der;
}

RbmSpin::VectorType RbmSpin::GetParameters() {
  VectorType pars(npar_);

  if (usea_) {
    pars.head(nv_) = a_;
  }

  if (useb_) {
    pars.segment(usea_ * nv_, nh_) = b_;
  }

  pars.tail(nv_ * nh_) = Eigen::Map<VectorType>(W_.data(), nv_ * nh_);

  return pars;
}

void RbmSpin::SetParameters(VectorConstRefType pars) {
  if (usea_) {
    a_ = pars.head(nv_);
  }

  if (useb_) {
    b_ = pars.segment(usea_ * nv_, nh_);
  }

  VectorType Wpars = pars.tail(nv_ * nh_);

  W_ = Eigen::Map<MatrixType>(Wpars.data(), nv_, nh_);
}

// Value of the logarithm of the wave-function
// using pre-computed look-up tables for efficiency
Complex RbmSpin::LogValSingle(VisibleConstType v, const any & /*unused*/) {
  return v.dot(a_) + SumLogCosh(W_.transpose() * v + b_);
}

void RbmSpin::Save(const std::string &filename) const {
  json state;
  state["Name"] = "RbmSpin";
  state["Nvisible"] = nv_;
  state["Nhidden"] = nh_;
  state["UseVisibleBias"] = usea_;
  state["UseHiddenBias"] = useb_;
  state["a"] = a_;
  state["b"] = b_;
  state["W"] = W_;
  WriteJsonToFile(state, filename);
}

void RbmSpin::Load(const std::string &filename) {
  auto const pars = ReadJsonFromFile(filename);
  std::string name = FieldVal<std::string>(pars, "Name");
  if (name != "RbmSpin") {
    throw InvalidInputError(
        "Error while constructing RbmSpin from input parameters");
  }

  if (FieldExists(pars, "Nvisible")) {
    nv_ = FieldVal<int>(pars, "Nvisible");
  }
  if (nv_ != GetHilbert().Size()) {
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
  if (FieldExists(pars, "a")) {
    a_ = FieldVal<VectorType>(pars, "a");
  } else {
    a_.setZero();
  }

  if (FieldExists(pars, "b")) {
    b_ = FieldVal<VectorType>(pars, "b");
  } else {
    b_.setZero();
  }
  if (FieldExists(pars, "W")) {
    W_ = FieldVal<MatrixType>(pars, "W");
  }
}

bool RbmSpin::IsHolomorphic() const noexcept { return true; }

}  // namespace netket
