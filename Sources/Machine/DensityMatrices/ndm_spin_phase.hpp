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

#ifndef NETKET_NDM_SPIN_PHASE_HPP
#define NETKET_NDM_SPIN_PHASE_HPP

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Machine/rbm_spin.hpp"
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"
#include "abstract_density_matrix.hpp"

namespace netket {

/** Neural Density Matrix machine class with spin 1/2 hidden units.
This version has real-valued weights and two NDMs parameterizing phase and
amplitude
 *
 */
class NdmSpinPhase : public AbstractDensityMatrix {
  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of ancillary units
  int na_;

  // number of parameters
  int npar_;

  // visible units bias
  RealVectorType b1_;
  RealVectorType b2_;

  // hidden units bias
  RealVectorType h1_;
  RealVectorType h2_;

  // ancillary units bias
  RealVectorType d1_;

  // hidden unit weights
  RealMatrixType W1_;
  RealMatrixType W2_;

  // ancillary unit weights
  RealMatrixType U1_;
  RealMatrixType U2_;

  // Caches
  RealVectorType thetas_r1_;
  RealVectorType thetas_r2_;
  RealVectorType thetas_c1_;
  RealVectorType thetas_c2_;
  RealVectorType lnthetas_r1_;
  RealVectorType lnthetas_r2_;
  RealVectorType lnthetas_c1_;
  RealVectorType lnthetas_c2_;
  RealVectorType thetasnew_r1_;
  RealVectorType thetasnew_r2_;
  RealVectorType thetasnew_c1_;
  RealVectorType thetasnew_c2_;
  RealVectorType lnthetasnew_r1_;
  RealVectorType lnthetasnew_r2_;
  RealVectorType lnthetasnew_c1_;
  RealVectorType lnthetasnew_c2_;

  RealVectorType thetas_a1_;
  RealVectorType thetas_a2_;
  RealVectorType thetasnew_a1_;
  RealVectorType thetasnew_a2_;
  VectorType pi_;
  VectorType lnpi_;
  VectorType lnpinew_;

  bool useb_;
  bool useh_;
  bool used_;

  const Complex I_;

 public:
  explicit NdmSpinPhase(std::shared_ptr<const AbstractHilbert> hilbert,
                        int nhidden = 0, int nancilla = 0, int alpha = 0,
                        int beta = 0, bool useb = true, bool useh = true,
                        bool used = true);

  void Init();

  int Nvisible() const override { return nv_; }

  int Nhidden() const { return nh_; }

  int Nancilla() const { return na_; }

  int Npar() const override { return npar_; }

  VectorType GetParameters() override;

  void SetParameters(VectorConstRefType pars) override;

  void InitRandomPars(int seed, double sigma) override;

  void InitLookup(VisibleConstType v_r, VisibleConstType v_c,
                  LookupType &lt) override;

  void UpdateLookup(VisibleConstType v_r, VisibleConstType v_c,
                    const std::vector<int> &tochange_r,
                    const std::vector<int> &tochange_c,
                    const std::vector<double> &newconf_r,
                    const std::vector<double> &newconf_c,
                    LookupType &lt) override;

  VectorType DerLog(VisibleConstType v_r, VisibleConstType v_c) override;

  VectorType DerLog(VisibleConstType v_r, VisibleConstType v_c,
                    const LookupType &lt) override;

  Complex LogVal(VisibleConstType v_r, VisibleConstType v_c) override;

  Complex LogVal(VisibleConstType v_r, VisibleConstType v_c,
                 const LookupType &lt) override;

  VectorType LogValDiff(
      VisibleConstType v_r, VisibleConstType v_c,
      const std::vector<std::vector<int>> &tochange_r,
      const std::vector<std::vector<int>> &tochange_c,
      const std::vector<std::vector<double>> &newconf_r,
      const std::vector<std::vector<double>> &newconf_c) override;

  Complex LogValDiff(VisibleConstType v_r, VisibleConstType v_c,
                     const std::vector<int> &tochange_r,
                     const std::vector<int> &tochange_c,
                     const std::vector<double> &newconf_r,
                     const std::vector<double> &newconf_c,
                     const LookupType &lt) override;

  inline static double lncosh(double x) {
    const double xp = std::abs(x);
    if (xp <= 12.) {
      return std::log(std::cosh(xp));
    } else {
      const static double log2v = std::log(2.);
      return xp - log2v;
    }
  }

  void Save(const std::string &filename) const override;

  void Load(const std::string &filename) override;

  bool IsHolomorphic() const noexcept override { return false; }
};

}  // namespace netket

#endif  // NETKET_NDM_SPIN_PHASE_HPP
