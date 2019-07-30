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

  inline void Init();

 public:
  explicit Jastrow(std::shared_ptr<const AbstractHilbert> hilbert);

  int Nvisible() const override;
  int Npar() const override;

  VectorType GetParameters() override;
  void SetParameters(VectorConstRefType pars) override;
  any InitLookup(VisibleConstType v) override;
  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf, any &lt) override;
  Complex LogValSingle(VisibleConstType v, const any &lt) override;

  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override;

  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const any &lt) override;

  VectorType DerLogSingle(VisibleConstType v, const any & /*unused*/) override;

  void Save(std::string const &filename) const override;
  void Load(std::string const &filename) override;

  bool IsHolomorphic() const noexcept override;
};

}  // namespace netket

#endif  // NETKET_JASTROW_HPP
