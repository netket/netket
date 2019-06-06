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

#ifndef NETKET_RBM_SPIN_REAL_HPP
#define NETKET_RBM_SPIN_REAL_HPP

#include "Machine/abstract_machine.hpp"

namespace netket {

using json = nlohmann::json;

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
and real-valued weights
 *
 */
class RbmSpinReal : public AbstractMachine {
  const AbstractHilbert &hilbert_;

  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of parameters
  int npar_;

  // weights
  RealMatrixType W_;

  // visible units bias
  RealVectorType a_;

  // hidden units bias
  RealVectorType b_;

  RealVectorType thetas_;
  RealVectorType lnthetas_;
  RealVectorType thetasnew_;
  RealVectorType lnthetasnew_;

  bool usea_;
  bool useb_;

 public:
  explicit RbmSpinReal(const AbstractHilbert &hilbert, int nhidden = 0,
                       int alpha = 0, bool usea = true, bool useb = true);

  inline void Init();

  virtual int Npar() const override;
  virtual int Nvisible() const override;
  /*constexpr*/ int Nhidden() const noexcept { return nh_; }

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

  virtual Complex LogVal(VisibleConstType v) override;
  virtual Complex LogVal(VisibleConstType v, const LookupType &lt) override;
  virtual VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override;
  virtual Complex LogValDiff(VisibleConstType v,
                             const std::vector<int> &tochange,
                             const std::vector<double> &newconf,
                             const LookupType &lt) override;

  virtual const AbstractHilbert &GetHilbert() const noexcept override;

  virtual void to_json(json &j) const override;
  virtual void from_json(const json &pars) override;

  virtual bool IsHolomorphic() override;
};

}  // namespace netket

#endif  // NETKET_RBM_SPIN_REAL_HPP
