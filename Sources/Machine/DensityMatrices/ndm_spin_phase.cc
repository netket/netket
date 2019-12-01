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

#include "ndm_spin_phase.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Machine/rbm_spin.hpp"
#include "Utils/all_utils.hpp"
#include "Utils/log_cosh.hpp"

namespace netket {

using RealVectorType = NdmSpinPhase::RealVectorType;
using VectorType = AbstractMachine::VectorType;

void NdmSpinPhase::Init() {
  W1_.resize(nv_, nh_);
  U1_.resize(nv_, na_);
  b1_.resize(nv_);
  h1_.resize(nh_);
  d1_.resize(na_);

  W2_.resize(nv_, nh_);
  U2_.resize(nv_, na_);
  b2_.resize(nv_);
  h2_.resize(nh_);

  BatchSize(1);

  npar_ = 2 * nv_ * (nh_ + na_);

  if (useb_) {
    npar_ += 2 * nv_;
  } else {
    b1_.setZero();
    b2_.setZero();
  }

  if (useh_) {
    npar_ += 2 * nh_;
  } else {
    h1_.setZero();
    h2_.setZero();
  }

  if (used_) {
    npar_ += na_;
  } else {
    d1_.setZero();
  }

  InfoMessage() << "Phase NDM Initizialized with nvisible = " << nv_
                << " and nhidden  = " << nh_ << " and nancilla = " << na_
                << std::endl;
  InfoMessage() << "Using visible   bias = " << useb_ << std::endl;
  InfoMessage() << "Using hidden    bias  = " << useh_ << std::endl;
  InfoMessage() << "Using ancillary bias  = " << used_ << std::endl;
}

Index NdmSpinPhase::BatchSize() const noexcept { return thetas_r1_.rows(); }

void NdmSpinPhase::BatchSize(Index batch_size) {
  if (batch_size <= 0) {
    std::ostringstream msg;
    msg << "invalid batch size: " << batch_size
        << "; expected a positive number";
    throw InvalidInputError{msg.str()};
  }
  if (batch_size != BatchSize()) {
    thetas_r1_.resize(batch_size, nh_);
    thetas_r2_.resize(batch_size, nh_);
    thetas_c1_.resize(batch_size, nh_);
    thetas_c2_.resize(batch_size, nh_);

    lnthetas_r1_.resize(batch_size, nh_);
    lnthetas_r2_.resize(batch_size, nh_);
    lnthetas_c1_.resize(batch_size, nh_);
    lnthetas_c2_.resize(batch_size, nh_);

    thetasnew_r1_.resize(batch_size, nh_);
    thetasnew_r2_.resize(batch_size, nh_);
    thetasnew_c1_.resize(batch_size, nh_);
    thetasnew_c2_.resize(batch_size, nh_);

    lnthetasnew_r1_.resize(batch_size, nh_);
    lnthetasnew_r2_.resize(batch_size, nh_);
    lnthetasnew_c1_.resize(batch_size, nh_);
    lnthetasnew_c2_.resize(batch_size, nh_);

    thetas_a_.resize(batch_size, na_);
    lnthetas_a_.resize(batch_size, na_);
    thetas_a1_.resize(batch_size, na_);
    thetas_a2_.resize(batch_size, na_);
    thetasnew_a1_.resize(batch_size, na_);
    thetasnew_a2_.resize(batch_size, na_);
    pi_.resize(batch_size, na_);
    lnpi_.resize(batch_size, na_);
    lnpinew_.resize(batch_size, na_);

    vsum_.resize(batch_size, nv_);
    vdelta_.resize(batch_size, nv_);
  }
}

VectorType NdmSpinPhase::DerLogSingle(VisibleConstType vr, VisibleConstType vc,
                                      const any & /*cache*/) {
  VectorType der(npar_);

  const int impar = (npar_ + na_ * used_) / 2;

  if (useb_) {
    der.head(nv_) = 0.5 * (vr + vc);
    der.segment(impar, nv_) = I_ * 0.5 * (vr - vc);
  }
  lnthetas_r1_ = (W1_.transpose() * vr + h1_).array().tanh();
  lnthetas_r2_ = (W2_.transpose() * vr + h2_).array().tanh();
  lnthetas_c1_ = (W1_.transpose() * vc + h1_).array().tanh();
  lnthetas_c2_ = (W2_.transpose() * vc + h2_).array().tanh();

  if (useh_) {
    der.segment(useb_ * nv_, nh_) = 0.5 * (lnthetas_r1_ + lnthetas_c1_);
    der.segment(impar + useb_ * nv_, nh_) =
        I_ * 0.5 * (lnthetas_r2_ - lnthetas_c2_);
  }

  thetas_a1_ = 0.5 * U1_.transpose() * (vr + vc) + d1_;
  thetas_a2_ = 0.5 * U2_.transpose() * (vr - vc);
  lnpi_ = ((0.5 * U1_.transpose() * (vr + vc) + d1_).array() +
           I_ * (0.5 * U2_.transpose() * (vr - vc)).array())
              .tanh();

  if (used_) {
    der.segment(useb_ * nv_ + useh_ * nh_, na_) = lnpi_;
  }

  const int initw_1 = nv_ * useb_ + nh_ * useh_ + na_ * used_;
  const int initw_2 = nv_ * useb_ + nh_ * useh_;

  MatrixType wder =
      0.5 * (vr * lnthetas_r1_.transpose() + vc * lnthetas_c1_.transpose());
  der.segment(initw_1, nv_ * nh_) =
      Eigen::Map<VectorType>(wder.data(), nv_ * nh_);

  wder = 0.5 * I_ *
         (vr * lnthetas_r2_.transpose() - vc * lnthetas_c2_.transpose());
  der.segment(impar + initw_2, nv_ * nh_) =
      Eigen::Map<VectorType>(wder.data(), nv_ * nh_);

  const int initu_1 = initw_1 + nv_ * nh_;
  const int initu_2 = initw_2 + nv_ * nh_;

  MatrixType uder = 0.5 * (vr + vc) * lnpi_.transpose();
  der.segment(initu_1, nv_ * na_) =
      Eigen::Map<VectorType>(uder.data(), nv_ * na_);

  uder = 0.5 * I_ * (vr - vc) * lnpi_.transpose();
  der.segment(impar + initu_2, nv_ * na_) =
      Eigen::Map<VectorType>(uder.data(), nv_ * na_);

  return der;
}

VectorType NdmSpinPhase::GetParameters() {
  VectorType pars(npar_);

  const int impar = (npar_ + na_ * used_) / 2;

  if (useb_) {
    pars.head(nv_) = b1_;
    pars.segment(impar, nv_) = b2_;
  }

  if (useh_) {
    pars.segment(nv_ * useb_, nh_) = h1_;
    pars.segment(impar + nv_ * useb_, nh_) = h2_;
  }

  if (used_) {
    pars.segment(useb_ * nv_ + useh_ * nh_, na_) = d1_;
  }

  const int initw_1 = nv_ * useb_ + nh_ * useh_ + na_ * used_;
  const int initw_2 = nv_ * useb_ + nh_ * useh_;

  pars.segment(initw_1, nv_ * nh_) =
      Eigen::Map<RealVectorType>(W1_.data(), nv_ * nh_);
  pars.segment(impar + initw_2, nv_ * nh_) =
      Eigen::Map<RealVectorType>(W2_.data(), nv_ * nh_);

  const int initu_1 = initw_1 + nv_ * nh_;
  const int initu_2 = initw_2 + nv_ * nh_;

  pars.segment(initu_1, nv_ * na_) =
      Eigen::Map<RealVectorType>(U1_.data(), nv_ * na_);
  pars.segment(impar + initu_2, nv_ * na_) =
      Eigen::Map<RealVectorType>(U2_.data(), nv_ * na_);

  return pars;
}

void NdmSpinPhase::SetParameters(VectorConstRefType pars) {
  const int impar = (npar_ + na_ * used_) / 2;

  if (useb_) {
    b1_ = pars.head(nv_).real();
    b2_ = pars.segment(impar, nv_).real();
  }

  if (useh_) {
    h1_ = pars.segment(useb_ * nv_, nh_).real();
    h2_ = pars.segment(impar + useb_ * nv_, nh_).real();
  }

  if (used_) {
    d1_ = pars.segment(useb_ * nv_ + useh_ * nh_, na_).real();
  }

  const int initw_1 = nv_ * useb_ + nh_ * useh_ + na_ * used_;
  const int initw_2 = nv_ * useb_ + nh_ * useh_;

  VectorType Wpars = pars.segment(initw_1, nv_ * nh_);
  W1_ = Eigen::Map<MatrixType>(Wpars.data(), nv_, nh_).real();

  Wpars = pars.segment(impar + initw_2, nv_ * nh_);
  W2_ = Eigen::Map<MatrixType>(Wpars.data(), nv_, nh_).real();

  const int initu_1 = initw_1 + nv_ * nh_;
  const int initu_2 = initw_2 + nv_ * nh_;

  VectorType Upars = pars.segment(initu_1, nv_ * na_);
  U1_ = Eigen::Map<MatrixType>(Upars.data(), nv_, na_).real();

  Upars = pars.segment(impar + initu_2, nv_ * na_);
  U2_ = Eigen::Map<MatrixType>(Upars.data(), nv_, na_).real();
}

// Value of the logarithm of the wave-function
// using pre-computed look-up tables for efficiency
Complex NdmSpinPhase::LogValSingle(VisibleConstType vr, VisibleConstType vc,
                                   const any &lookup) {
  auto r1s = SumLogCosh(W1_.transpose() * vr + h1_);
  auto r2s = SumLogCosh(W2_.transpose() * vr + h2_);
  auto c1s = SumLogCosh(W1_.transpose() * vc + h1_);
  auto c2s = SumLogCosh(W2_.transpose() * vc + h2_);

  thetas_a1_ = 0.5 * U1_.transpose() * (vr + vc) + d1_;
  thetas_a2_ = 0.5 * U2_.transpose() * (vr - vc);

  auto lnpis = SumLogCosh(thetas_a1_ + I_ * thetas_a2_);

  auto gamma_1 = 0.5 * (r1s + c1s + (vr + vc).dot(b1_));

  auto gamma_2 = 0.5 * (r2s - c2s + (vr - vc).dot(b2_));

  return (gamma_1 + I_ * gamma_2 + lnpis);
}

void NdmSpinPhase::LogVal(Eigen::Ref<const RowMatrix<double>> vr,
                          Eigen::Ref<const RowMatrix<double>> vc,
                          Eigen::Ref<VectorType> out, const any &lup) {
  CheckShape(__FUNCTION__, "vr", {vr.rows(), vr.cols()},
             {vc.rows(), NvisiblePhysical()});
  CheckShape(__FUNCTION__, "vc", {vc.rows(), vc.cols()},
             {vr.rows(), NvisiblePhysical()});
  CheckShape(__FUNCTION__, "out", out.size(), vr.rows());

  BatchSize(vr.rows());

  vsum_ = vr + vc;
  vdelta_ = vr - vc;

  thetas_r1_ = (vr * W1_).rowwise() + h1_.transpose();
  thetas_r2_ = (vr * W2_).rowwise() + h2_.transpose();
  thetas_c1_ = (vc * W1_).rowwise() + h1_.transpose();
  thetas_c2_ = (vc * W2_).rowwise() + h2_.transpose();
  thetas_a_ =
      (0.5 * (vsum_ * U1_ + I_ * vdelta_ * U2_)).rowwise() + d1_.transpose();

  out.noalias() = (vsum_ * b1_ + I_ * vdelta_ * b2_);

#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    out(j) += SumLogCosh(thetas_r1_.row(j));
  }
#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    out(j) += I_ * SumLogCosh(thetas_r2_.row(j));
  }
#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    out(j) += SumLogCosh(thetas_c1_.row(j));
  }
#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    out(j) -= I_ * SumLogCosh(thetas_c2_.row(j));
  }

  // All previous term are multiplied by 0.5
  out = out * 0.5;

#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    out(j) += SumLogCosh(thetas_a_.row(j));
  }
}

void NdmSpinPhase::DerLog(Eigen::Ref<const RowMatrix<double>> vr,
                          Eigen::Ref<const RowMatrix<double>> vc,
                          Eigen::Ref<RowMatrix<Complex>> out,
                          const any &cache) {
  CheckShape(__FUNCTION__, "vr", {vr.rows(), vr.cols()},
             {vc.rows(), NvisiblePhysical()});
  CheckShape(__FUNCTION__, "vc", {vc.rows(), vc.cols()},
             {vr.rows(), NvisiblePhysical()});
  CheckShape(__FUNCTION__, "out", {out.rows(), out.cols()},
             {vr.rows(), Npar()});
  BatchSize(vr.rows());

  vsum_ = vr + vc;
  vdelta_ = vr - vc;

  const int impar = (npar_ + na_ * used_) / 2;

  auto i = Index{0};
  auto i2 = Index{impar};
  if (useb_) {
    out.block(0, i, BatchSize(), nv_) = 0.5 * vsum_;
    out.block(0, impar, BatchSize(), nv_) = I_ * 0.5 * vdelta_;
    i += nv_;
    i2 += nv_;
  }

  thetas_r1_ = ((vr * W1_).rowwise() + h1_.transpose()).array().tanh();
  thetas_r2_ = ((vr * W2_).rowwise() + h2_.transpose()).array().tanh();
  thetas_c1_ = ((vc * W1_).rowwise() + h1_.transpose()).array().tanh();
  thetas_c2_ = ((vc * W2_).rowwise() + h2_.transpose()).array().tanh();
  thetas_a_ =
      ((0.5 * (vsum_ * U1_ + I_ * vdelta_ * U2_)).rowwise() + d1_.transpose())
          .array()
          .tanh();
  if (useh_) {
    out.block(0, i, BatchSize(), nh_) = 0.5 * (thetas_r1_ + thetas_c1_);
    out.block(0, i2, BatchSize(), nh_) = I_ * 0.5 * (thetas_r2_ - thetas_c2_);
    i += nh_;
    i2 += nh_;
  }

  if (used_) {
    out.block(0, i, BatchSize(), na_) = thetas_a_;
    i += na_;
  }

  // TODO: Rewrite all those using tensors

#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    Eigen::Map<Eigen::MatrixXcd>{&out(j, i), W1_.rows(), W1_.cols()}.noalias() =
        0.5 * (vr.row(j).transpose() * thetas_r1_.row(j) +
               vc.row(j).transpose() * thetas_c1_.row(j));
  }

#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    Eigen::Map<Eigen::MatrixXcd>{&out(j, i2), W1_.rows(), W1_.cols()}
        .noalias() = 0.5 * I_ *
                     (vr.row(j).transpose() * thetas_r2_.row(j) -
                      vc.row(j).transpose() * thetas_c2_.row(j));
  }

  i += nv_ * nh_;
  i2 += nv_ * nh_;

  // TODO: Rewrite all those using tensors

#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    Eigen::Map<Eigen::MatrixXcd>{&out(j, i), U1_.rows(), U1_.cols()}.noalias() =
        0.5 * vsum_.row(j).transpose() * thetas_a_.row(j);
  }

#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    Eigen::Map<Eigen::MatrixXcd>{&out(j, i2), U2_.rows(), U2_.cols()}
        .noalias() = 0.5 * I_ * vdelta_.row(j).transpose() * thetas_a_.row(j);
  }
}

void NdmSpinPhase::Save(const std::string &filename) const {
  json state;
  state["Name"] = "NdmSpinPhase";
  state["Nvisible"] = nv_;
  state["Nhidden"] = nh_;
  state["Nancilla"] = na_;
  state["UseVisibleBias"] = useb_;
  state["UseHiddenBias"] = useh_;
  state["UseAncillaBias"] = used_;
  state["b1"] = b1_;
  state["h1"] = h1_;
  state["d1"] = d1_;
  state["W1"] = W1_;
  state["U1"] = U1_;

  state["b2"] = b2_;
  state["h2"] = h2_;
  state["W2"] = W2_;
  state["U2"] = U2_;
  WriteJsonToFile(state, filename);
}

void NdmSpinPhase::Load(const std::string &filename) {
  auto pars = ReadJsonFromFile(filename);
  std::string name = FieldVal<std::string>(pars, "Name");
  if (name != "NdmSpinPhase") {
    throw InvalidInputError(
        "Error while constructing RbmSpinPhase from input parameters");
  }

  if (FieldExists(pars, "Nvisible")) {
    nv_ = FieldVal<int>(pars, "Nvisible");
  }
  if (nv_ != GetHilbertPhysical().Size()) {
    throw InvalidInputError(
        "Number of visible units is incompatible with given "
        "Hilbert space");
  }

  if (FieldExists(pars, "Nhidden")) {
    nh_ = FieldVal<int>(pars, "Nhidden");
  } else {
    nh_ = nv_ * double(FieldVal<double>(pars, "Alpha"));
  }

  if (FieldExists(pars, "Nancilla")) {
    na_ = FieldVal<int>(pars, "Nancilla");
  } else {
    na_ = nv_ * double(FieldVal<double>(pars, "Beta"));
  }

  useb_ = FieldOrDefaultVal(pars, "UseVisibleBias", true);
  useh_ = FieldOrDefaultVal(pars, "UseHiddenBias", true);
  used_ = FieldOrDefaultVal(pars, "UseAncillaBias", true);

  Init();

  // Loading parameters, if defined in the input
  if (FieldExists(pars, "b1")) {
    b1_ = FieldVal<RealVectorType>(pars, "b1");
    b2_ = FieldVal<RealVectorType>(pars, "b2");
  } else {
    b1_.setZero();
    b2_.setZero();
  }

  if (FieldExists(pars, "h1")) {
    h1_ = FieldVal<RealVectorType>(pars, "h1");
    h2_ = FieldVal<RealVectorType>(pars, "h2");
  } else {
    h1_.setZero();
    h2_.setZero();
  }

  if (FieldExists(pars, "d1")) {
    d1_ = FieldVal<RealVectorType>(pars, "d1");
  } else {
    d1_.setZero();
  }

  if (FieldExists(pars, "W1")) {
    W1_ = FieldVal<RealMatrixType>(pars, "W1");
    W2_ = FieldVal<RealMatrixType>(pars, "W2");
  }

  if (FieldExists(pars, "U1")) {
    U1_ = FieldVal<RealMatrixType>(pars, "U1");
    U2_ = FieldVal<RealMatrixType>(pars, "U2");
  }
}

bool NdmSpinPhase::IsHolomorphic() const noexcept { return false; }
};  // namespace netket
