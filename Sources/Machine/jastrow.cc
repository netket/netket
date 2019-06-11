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

  thetas_.resize(nv_);
  thetasnew_.resize(nv_);

  InfoMessage() << "Jastrow WF Initizialized with nvisible = " << nv_
                << " and nparams = " << npar_ << std::endl;
}

int Jastrow::Nvisible() const { return nv_; }

int Jastrow::Npar() const { return npar_; }

void Jastrow::InitRandomPars(int seed, double sigma) {
  VectorType par(npar_);

  netket::RandomGaussian(par, seed, sigma);

  SetParameters(par);
}

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

void Jastrow::InitLookup(VisibleConstType v, LookupType &lt) {
  if (lt.VectorSize() == 0) {
    lt.AddVector(v.size());
  }
  if (lt.V(0).size() != v.size()) {
    lt.V(0).resize(v.size());
  }

  lt.V(0) = (W_.transpose() * v);  // does not matter the transpose W is symm
}

// same as for the RBM
void Jastrow::UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                           const std::vector<double> &newconf, LookupType &lt) {
  if (tochange.size() != 0) {
    for (std::size_t s = 0; s < tochange.size(); s++) {
      const int sf = tochange[s];
      lt.V(0) += W_.row(sf) * (newconf[s] - v(sf));
    }
  }
}

Complex Jastrow::LogVal(VisibleConstType v) { return 0.5 * v.dot(W_ * v); }

// Value of the logarithm of the wave-function
// using pre-computed look-up tables for efficiency
Complex Jastrow::LogVal(VisibleConstType v, const LookupType &lt) {
  return 0.5 * v.dot(lt.V(0));
}

// Difference between logarithms of values, when one or more visible variables
// are being flipped
Jastrow::VectorType Jastrow::LogValDiff(
    VisibleConstType v, const std::vector<std::vector<int>> &tochange,
    const std::vector<std::vector<double>> &newconf) {
  const std::size_t nconn = tochange.size();
  VectorType logvaldiffs = VectorType::Zero(nconn);

  thetas_ = (W_.transpose() * v);
  Complex logtsum = 0.5 * v.dot(thetas_);

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

Complex Jastrow::LogValDiff(VisibleConstType v,
                            const std::vector<int> &tochange,
                            const std::vector<double> &newconf,
                            const LookupType &lt) {
  Complex logvaldiff = 0.;

  if (tochange.size() != 0) {
    Complex logtsum = 0.5 * v.dot(lt.V(0));
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

Jastrow::VectorType Jastrow::DerLog(VisibleConstType v) {
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

void Jastrow::to_json(json &j) const {
  j["Name"] = "Jastrow";
  j["Nvisible"] = nv_;
  j["W"] = W_;
}

void Jastrow::from_json(const json &pars) {
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

}  // namespace netket
