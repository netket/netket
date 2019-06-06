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

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <vector>
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"
#include "rbm_spin.hpp"

#ifndef NETKET_RBM_MULTIVAL_HPP
#define NETKET_RBM_MULTIVAL_HPP

#include "abstract_machine.hpp"

namespace netket {

// Restricted Boltzman Machine wave function
// for generic (finite) local hilbert space
class RbmMultival : public AbstractMachine {
  const AbstractHilbert &hilbert_;

  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of parameters
  int npar_;

  // local size of hilbert space
  int ls_;

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

  Eigen::VectorXd localconfs_;
  Eigen::MatrixXd mask_;

  Eigen::VectorXd vtilde_;

  std::map<double, int> confindex_;

 public:
  explicit RbmMultival(const AbstractHilbert &hilbert, int nhidden = 0,
                       int alpha = 0, bool usea = true, bool useb = true);

  virtual int Npar() const override;
  virtual int Nvisible() const override;
  /*constexpr*/ int Nhidden() const noexcept { return nh_; }

  virtual const AbstractHilbert &GetHilbert() const noexcept override;

  virtual void InitRandomPars(int seed, double sigma) override;
  virtual void InitLookup(VisibleConstType v, LookupType &lt) override;
  virtual void UpdateLookup(VisibleConstType v,
                            const std::vector<int> &tochange,
                            const std::vector<double> &newconf,
                            LookupType &lt) override;

  virtual VectorType DerLog(VisibleConstType v) override;
  virtual VectorType DerLog(VisibleConstType v, const LookupType &lt) override;

  virtual VectorType GetParameters() override;
  virtual void SetParameters(VectorConstRefType pars) override;

  // Value of the logarithm of the wave-function
  virtual Complex LogVal(VisibleConstType v) override;
  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  virtual Complex LogVal(VisibleConstType v, const LookupType &lt) override;
  // Difference between logarithms of values, when one or more visible variables
  // are being changed
  virtual VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override;
  // Difference between logarithms of values, when one or more visible variables
  // are being changed Version using pre-computed look-up tables for efficiency
  // on a small number of local changes
  virtual Complex LogValDiff(VisibleConstType v,
                             const std::vector<int> &tochange,
                             const std::vector<double> &newconf,
                             const LookupType &lt) override;

 private:
  inline void Init();

  // Computhes the values of the theta pseudo-angles
  inline void ComputeTheta(VisibleConstType v, VectorType &theta) {
    ComputeVtilde(v, vtilde_);
    theta = (W_.transpose() * vtilde_ + b_);
  }

  inline void ComputeVtilde(VisibleConstType v, Eigen::VectorXd &vtilde) {
    auto t = (localconfs_.array() == (mask_ * v).array());
    vtilde = t.template cast<double>();
  }

  virtual void to_json(json &j) const override;
  virtual void from_json(const json &pars) override;
};

}  // namespace netket

#endif
