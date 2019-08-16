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

#include "rbm_spin_real.hpp"

#include "Machine/rbm_spin.hpp"
#include "Utils/json_utils.hpp"
#include "Utils/log_cosh.hpp"
#include "Utils/messages.hpp"

namespace netket {

RbmSpinReal::RbmSpinReal(std::shared_ptr<const AbstractHilbert> hilbert,
                         int nhidden, int alpha, bool usea, bool useb)
    : AbstractMachine(hilbert), nv_(hilbert->Size()), usea_(usea), useb_(useb) {
  nh_ = std::max(nhidden, alpha * nv_);
  Init();
}

void RbmSpinReal::Init() {
  W_.resize(nv_, nh_);
  a_.resize(nv_);
  b_.resize(nh_);

  lnthetas_.resize(nh_);

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

  InfoMessage() << "Real-valued RBM Initizialized with nvisible = " << nv_
                << " and nhidden = " << nh_ << std::endl;
  InfoMessage() << "Using visible bias = " << usea_ << std::endl;
  InfoMessage() << "Using hidden bias  = " << useb_ << std::endl;
}

int RbmSpinReal::Nvisible() const { return nv_; }

int RbmSpinReal::Npar() const { return npar_; }

RbmSpinReal::VectorType RbmSpinReal::DerLogSingle(VisibleConstType v,
                                                  const any &) {
  VectorType der(npar_);

  if (usea_) {
    der.head(nv_) = v;
  }

  lnthetas_ = (W_.transpose() * v + b_).array().real().tanh();

  if (useb_) {
    der.segment(usea_ * nv_, nh_) = lnthetas_;
  }

  MatrixType wder = (v * lnthetas_.transpose());
  der.tail(nv_ * nh_) = Eigen::Map<VectorType>(wder.data(), nv_ * nh_);

  return der;
}

RbmSpinReal::VectorType RbmSpinReal::GetParameters() {
  VectorType pars(npar_);

  if (usea_) {
    pars.head(nv_) = a_;
  }

  if (useb_) {
    pars.segment(usea_ * nv_, nh_) = b_;
  }

  pars.tail(nv_ * nh_) = Eigen::Map<RealVectorType>(W_.data(), nv_ * nh_);

  return pars;
}

void RbmSpinReal::SetParameters(VectorConstRefType pars) {
  if (usea_) {
    a_ = pars.head(nv_).real();
  }

  if (useb_) {
    b_ = pars.segment(usea_ * nv_, nh_).real();
  }

  VectorType Wpars = pars.tail(nv_ * nh_);

  W_ = Eigen::Map<MatrixType>(Wpars.data(), nv_, nh_).real();
}

// Value of the logarithm of the wave-function
// using pre-computed look-up tables for efficiency
Complex RbmSpinReal::LogValSingle(VisibleConstType v, const any &) {
  return (v.dot(a_) + SumLogCosh(W_.transpose() * v + b_));
}

void RbmSpinReal::LogVal(Eigen::Ref<const RowMatrix<double>> x,
                         Eigen::Ref<Eigen::VectorXcd> out, const any &) {
  CheckShape(__FUNCTION__, "v", {x.rows(), x.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "out", out.size(), x.rows());

  SumLogCosh((x * W_).rowwise() + b_.transpose(), out);

  if (usea_) {
    out += x * a_;
  }
}

void RbmSpinReal::Save(const std::string &filename) const {
  json state;
  state["Name"] = "RbmSpinReal";
  state["Nvisible"] = nv_;
  state["Nhidden"] = nh_;
  state["UseVisibleBias"] = usea_;
  state["UseHiddenBias"] = useb_;
  state["a"] = a_;
  state["b"] = b_;
  state["W"] = W_;
  WriteJsonToFile(state, filename);
}

void RbmSpinReal::Load(const std::string &filename) {
  auto const pars = ReadJsonFromFile(filename);
  std::string name = FieldVal<std::string>(pars, "Name");
  if (name != "RbmSpinReal") {
    throw InvalidInputError(
        "Error while constructing RbmSpinReal from input parameters");
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
    a_ = FieldVal<RealVectorType>(pars, "a");
  } else {
    a_.setZero();
  }

  if (FieldExists(pars, "b")) {
    b_ = FieldVal<RealVectorType>(pars, "b");
  } else {
    b_.setZero();
  }
  if (FieldExists(pars, "W")) {
    W_ = FieldVal<RealMatrixType>(pars, "W");
  }
}

bool RbmSpinReal::IsHolomorphic() const noexcept { return false; }

}  // namespace netket
