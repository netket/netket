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

#ifndef NETKET_RBM_MULTIVAL_HPP
#define NETKET_RBM_MULTIVAL_HPP

#include <map>
#include <vector>

#include <Eigen/Dense>

#include "Machine/abstract_machine.hpp"
#include "Machine/rbm_spin.hpp"

namespace netket {

// Restricted Boltzman Machine wave function
// for generic (finite) local hilbert space
class RbmMultival : public AbstractMachine {
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
  explicit RbmMultival(std::shared_ptr<const AbstractHilbert> hilbert,
                       int nhidden = 0, int alpha = 0, bool usea = true,
                       bool useb = true);

  int Npar() const override;
  int Nvisible() const override;
  /*constexpr*/ int Nhidden() const noexcept { return nh_; }

  void InitRandomPars(int seed, double sigma) override;
  void InitLookup(VisibleConstType v, LookupType &lt) override;
  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override;

  VectorType DerLog(VisibleConstType v) override;
  VectorType DerLog(VisibleConstType v, const LookupType &lt) override;

  VectorType GetParameters() override;
  void SetParameters(VectorConstRefType pars) override;

  // Value of the logarithm of the wave-function
  Complex LogVal(VisibleConstType v) override;
  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  Complex LogVal(VisibleConstType v, const LookupType &lt) override;
  // Difference between logarithms of values, when one or more visible variables
  // are being changed
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override;
  // Difference between logarithms of values, when one or more visible variables
  // are being changed Version using pre-computed look-up tables for efficiency
  // on a small number of local changes
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override;

  void to_json(json &j) const override;
  void from_json(const json &pars) override;

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
};

}  // namespace netket

#endif
