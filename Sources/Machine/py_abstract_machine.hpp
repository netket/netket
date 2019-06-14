// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#ifndef NETKET_MACHINE_PY_ABSTRACT_MACHINE_HPP
#define NETKET_MACHINE_PY_ABSTRACT_MACHINE_HPP

#include "Machine/abstract_machine.hpp"

namespace netket {

class PyAbstractMachine : public AbstractMachine {
 public:
  PyAbstractMachine(std::shared_ptr<const AbstractHilbert> hilbert)
      : AbstractMachine{std::move(hilbert)} {}

  int Npar() const override;
  int Nvisible() const override;
  bool IsHolomorphic() const noexcept override;

  VectorType GetParameters() override;
  void SetParameters(VectorConstRefType pars) override;
  void InitRandomPars(int seed, double sigma) override;

  Complex LogVal(VisibleConstType v) override;
  Complex LogVal(VisibleConstType v, const LookupType & /*unused*/) override;

  void InitLookup(VisibleConstType /*unused*/,
                  LookupType & /*unused*/) override;
  void UpdateLookup(VisibleConstType /*unused*/,
                    const std::vector<int> & /*unused*/,
                    const std::vector<double> & /*unused*/,
                    LookupType & /*unused*/) override;

  VectorType LogValDiff(
      VisibleConstType old_v, const std::vector<std::vector<int>> &to_change,
      const std::vector<std::vector<double>> &new_conf) override;
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &to_change,
                     const std::vector<double> &new_conf,
                     const LookupType & /*unused*/) override;

  VectorType DerLog(VisibleConstType v) override;
  VectorType DerLog(VisibleConstType v, const LookupType & /*lt*/) override;
  VectorType DerLogChanged(VisibleConstType old_v,
                           const std::vector<int> &to_change,
                           const std::vector<double> &new_conf) override;

  void Save(const std::string &filename) const override;
  void Load(const std::string &filename) override;

  ~PyAbstractMachine() override = default;

 private:
  inline Complex LogValDiff(VisibleConstType old_v,
                            const std::vector<int> &to_change,
                            const std::vector<double> &new_conf);
};

}  // namespace netket

#endif  // NETKET_MACHINE_PY_ABSTRACT_MACHINE_HPP
