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

#include "Machine/jastrow.hpp"

#include "Utils/json_utils.hpp"
#include "Utils/messages.hpp"

namespace netket {

Jastrow::Jastrow(std::shared_ptr<const AbstractHilbert> hilbert)
    : AbstractMachine(hilbert), nv_(hilbert->Size()) {
  Init();
}

void Jastrow::Init() {
  if (nv_ < 2) {
    throw InvalidInputError(
        "Cannot construct Jastrow states with less than two visible units");
  }

  W_.resize(nv_, nv_);
  W_.setZero();

  npar_ = (nv_ * (nv_ - 1)) / 2;

  InfoMessage() << "Jastrow WF Initizialized with nvisible = " << nv_
                << " and nparams = " << npar_ << std::endl;
}

int Jastrow::Nvisible() const { return nv_; }

int Jastrow::Npar() const { return npar_; }

Jastrow::VectorType Jastrow::GetParameters() {
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

void Jastrow::SetParameters(VectorConstRefType pars) {
  int k = 0;

  for (int i = 0; i < nv_; i++) {
    W_(i, i) = Complex(0.);
    for (int j = i + 1; j < nv_; j++) {
      W_(i, j) = pars(k);
      W_(j, i) = W_(i, j);  // create the lower triangle
      k++;
    }
  }
}

// Value of the logarithm of the wave-function on a batched x
void Jastrow::LogVal(Eigen::Ref<const RowMatrix<double>> x,
                     Eigen::Ref<Eigen::VectorXcd> out, const any &) {
  CheckShape(__FUNCTION__, "v", {x.rows(), x.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "out", out.size(), x.rows());

  out = 0.5 * (x * W_).cwiseProduct(x).rowwise().sum();
}

Complex Jastrow::LogValSingle(VisibleConstType x, const any &) {
  return 0.5 * x.dot(W_ * x);
}

Jastrow::VectorType Jastrow::DerLogSingle(VisibleConstType v,
                                          const any & /*unused*/) {
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

void Jastrow::Save(std::string const &filename) const {
  json state;
  state["Name"] = "Jastrow";
  state["Nvisible"] = nv_;
  state["W"] = W_;
  WriteJsonToFile(state, filename);
}

void Jastrow::Load(const std::string &filename) {
  auto const pars = ReadJsonFromFile(filename);
  if (pars.at("Name") != "Jastrow") {
    throw InvalidInputError("Error while constructing Jastrow from Json input");
  }

  if (FieldExists(pars, "Nvisible")) {
    nv_ = pars["Nvisible"];
  }
  if (nv_ != GetHilbert().Size()) {
    throw InvalidInputError(
        "Number of visible units is incompatible with given "
        "Hilbert space");
  }

  Init();

  if (FieldExists(pars, "W")) {
    W_ = pars["W"];
  }
}

bool Jastrow::IsHolomorphic() const noexcept { return true; }

}  // namespace netket
