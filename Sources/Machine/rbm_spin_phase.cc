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

#include "rbm_spin_phase.hpp"

#include "Machine/rbm_spin.hpp"
#include "Utils/json_utils.hpp"
#include "Utils/log_cosh.hpp"
#include "Utils/messages.hpp"

namespace netket {

RbmSpinPhase::RbmSpinPhase(std::shared_ptr<const AbstractHilbert> hilbert,
                           int nhidden, int alpha, bool usea, bool useb)
    : AbstractMachine(hilbert),
      nv_(hilbert->Size()),
      usea_(usea),
      useb_(useb),
      I_(0, 1) {
  nh_ = std::max(nhidden, alpha * nv_);
  Init();
}

void RbmSpinPhase::Init() {
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

int RbmSpinPhase::Nvisible() const { return nv_; }

int RbmSpinPhase::Npar() const { return npar_; }

RbmSpinPhase::VectorType RbmSpinPhase::DerLogSingle(VisibleConstType v,
                                                    const any & /*lookup*/) {
  VectorType der(npar_);

  const int impar = npar_ / 2;

  if (usea_) {
    der.head(nv_) = v;
    der.segment(impar, nv_) = I_ * v;
  }

  lnthetas1_ = (W1_.transpose() * v + b1_).array().tanh();
  lnthetas2_ = (W2_.transpose() * v + b2_).array().tanh();

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

RbmSpinPhase::VectorType RbmSpinPhase::GetParameters() {
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

void RbmSpinPhase::SetParameters(VectorConstRefType pars) {
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
// using pre-computed look-up tables for efficiency
Complex RbmSpinPhase::LogValSingle(VisibleConstType v, const any & /*lookup*/) {
  return (v.dot(a1_) + SumLogCosh(W1_.transpose() * v + b1_) +
          I_ * (v.dot(a2_) + SumLogCosh(W2_.transpose() * v + b2_)));
}

void RbmSpinPhase::LogVal(Eigen::Ref<const RowMatrix<double>> x,
                          Eigen::Ref<Eigen::VectorXcd> out, const any &) {
  CheckShape(__FUNCTION__, "v", {x.rows(), x.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "out", out.size(), x.rows());

  SumLogCoshReIm((x * W1_).rowwise() + b1_.transpose(),
                 (x * W2_).rowwise() + b2_.transpose(), out);

  if (usea_) {
    out.real() += x * a1_;
    out.imag() += x * a2_;
  }
}

void RbmSpinPhase::Save(const std::string &filename) const {
  json state;
  state["Name"] = "RbmSpinPhase";
  state["Nvisible"] = nv_;
  state["Nhidden"] = nh_;
  state["UseVisibleBias"] = usea_;
  state["UseHiddenBias"] = useb_;
  state["a1"] = a1_;
  state["b1"] = b1_;
  state["W1"] = W1_;
  state["a2"] = a2_;
  state["b2"] = b2_;
  state["W2"] = W2_;
  WriteJsonToFile(state, filename);
}

void RbmSpinPhase::Load(const std::string &filename) {
  auto const pars = ReadJsonFromFile(filename);
  std::string name = FieldVal<std::string>(pars, "Name");
  if (name != "RbmSpinPhase") {
    throw InvalidInputError(
        "Error while constructing RbmSpinPhase from input parameters");
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

bool RbmSpinPhase::IsHolomorphic() const noexcept { return false; }

}  // namespace netket
