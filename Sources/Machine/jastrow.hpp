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

#ifndef NETKET_JASTROW_HPP
#define NETKET_JASTROW_HPP

#include "Machine/abstract_machine.hpp"

namespace netket {

/** Jastrow machine class.
 *
 */
class Jastrow : public AbstractMachine {
  // number of visible units
  int nv_;

  // number of parameters
  int npar_;

  // weights
  MatrixType W_;

  // buffers
  VectorType thetas_;
  VectorType thetasnew_;

 public:
  explicit Jastrow(std::shared_ptr<const AbstractHilbert> hilbert);

  inline void Init();

  int Nvisible() const override;
  int Npar() const override;

  void InitRandomPars(int seed, double sigma) override;
  VectorType GetParameters() override;
  void SetParameters(VectorConstRefType pars) override;
  void InitLookup(VisibleConstType v, LookupType &lt) override;
  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override;
  Complex LogVal(VisibleConstType v) override;
  Complex LogVal(VisibleConstType v, const LookupType &lt) override;

  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override;

  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override;

  VectorType DerLog(VisibleConstType v) override;

  void to_json(json &j) const override;
  void from_json(const json &pars) override;
};

}  // namespace netket

#endif  // NETKET_JASTROW_HPP
