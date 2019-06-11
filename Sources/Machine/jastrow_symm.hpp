// Copyright 2018 The Simons Foundation, Inc. - All
// Rights Reserved.
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

#ifndef NETKET_JAS_SYMM_HPP
#define NETKET_JAS_SYMM_HPP

#include "Machine/abstract_machine.hpp"

namespace netket {

// Jastrow with permutation symmetries
class JastrowSymm : public AbstractMachine {
  const AbstractGraph &graph_;

  std::vector<std::vector<int>> permtable_;
  int permsize_;

  // number of visible units
  int nv_;

  // number of parameters
  int npar_;

  // number of parameters without symmetries
  int nbarepar_;

  // weights
  MatrixType W_;

  // weights with symmetries
  MatrixType Wsymm_;

  VectorType thetas_;
  VectorType thetasnew_;

  Eigen::MatrixXd DerMatSymm_;
  Eigen::MatrixXi Wtemp_;

 public:
  explicit JastrowSymm(std::shared_ptr<const AbstractHilbert> hilbert);

  int Nvisible() const override;
  int Npar() const override;

  void InitRandomPars(int seed, double sigma) override;
  void InitLookup(VisibleConstType v, LookupType &lt) override;
  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override;

  VectorType DerLog(VisibleConstType v) override;

  VectorType GetParameters() override;
  void SetParameters(VectorConstRefType pars) override;

  Complex LogVal(VisibleConstType v) override;
  Complex LogVal(VisibleConstType v, const LookupType &lt) override;
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override;
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override;

  void to_json(json &j) const override;
  void from_json(const json &pars) override;

 private:
  inline void Init(const AbstractGraph &graph);

  VectorType BareDerLog(VisibleConstType v);
  void SetBareParameters();
};

}  // namespace netket

#endif  // NETKET_JAS_SYMM_HPP
