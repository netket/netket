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

#ifndef NETKET_MACHINE_HPP
#define NETKET_MACHINE_HPP

#include <fstream>
#include <memory>

#include "Graph/graph.hpp"
#include "Operator/hamiltonian.hpp"
#include "abstract_machine.hpp"
#include "ffnn.hpp"
#include "jastrow.hpp"
#include "jastrow_symm.hpp"
#include "mps_periodic.hpp"
#include "rbm_multival.hpp"
#include "rbm_spin.hpp"
#include "rbm_spin_symm.hpp"

namespace netket {
// TODO remove
template <class T>
class Machine : public AbstractMachine<T> {
  std::unique_ptr<AbstractMachine<T>> m_;

  const AbstractHilbert &hilbert_;

 public:
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  template <class Partype>
  explicit Machine(const AbstractHilbert &hilbert, const Partype &pars)
      : hilbert_(hilbert) {
    const auto pconv = ParsConv(pars);
    Init(hilbert_, pconv);
    InitParameters(pconv);
  }

  template <class Partype>
  explicit Machine(const Hamiltonian &hamiltonian, const Partype &pars)
      : hilbert_(hamiltonian.GetHilbert()) {
    const auto pconv = ParsConv(pars);
    Init(hilbert_, pconv);
    InitParameters(pconv);
  }

  template <class Partype>
  explicit Machine(const AbstractGraph &graph, const AbstractHilbert &hilbert,
                   const Partype &pars)
      : hilbert_(hilbert) {
    const auto pconv = ParsConv(pars);
    Init(hilbert_, pconv);
    Init(graph, hilbert, pconv);
    InitParameters(pconv);
  }

  template <class Partype>
  explicit Machine(const AbstractGraph &graph, const Hamiltonian &hamiltonian,
                   const Partype &pars)
      : hilbert_(hamiltonian.GetHilbert()) {
    const auto pconv = ParsConv(pars);
    Init(hilbert_, pconv);
    Init(graph, hilbert_, pconv);
    InitParameters(pconv);
  }

  template <class Partype>
  void Init(const AbstractHilbert &hilbert, const Partype &pars) {
    CheckInput(pars);

    std::string name = FieldVal<std::string>(pars, "Name");
    if (name == "RbmSpin") {
      m_ = netket::make_unique<RbmSpin<T>>(hilbert, pars);
    } else if (name == "RbmMultival") {
      m_ = netket::make_unique<RbmMultival<T>>(hilbert, pars);
    } else if (name == "Jastrow") {
      m_ = netket::make_unique<Jastrow<T>>(hilbert, pars);
    } else if (name == "MPSperiodic") {
      if (FieldExists(pars, "Diagonal") && pars["Diagonal"]) {
        m_ = netket::make_unique<MPSPeriodic<T, true>>(hilbert, pars);
      } else {
        m_ = netket::make_unique<MPSPeriodic<T, false>>(hilbert, pars);
      }
    }
  }
  template <class Partype>
  void Init(const AbstractGraph &graph, const AbstractHilbert &hilbert,
            const Partype &pars) {
    CheckInput(pars);
    std::string name = FieldVal<std::string>(pars, "Name");
    if (name == "RbmSpinSymm") {
      m_ = netket::make_unique<RbmSpinSymm<T>>(graph, hilbert, pars);
    } else if (name == "FFNN") {
      m_ = netket::make_unique<FFNN<T>>(graph, hilbert, pars);
    } else if (name == "JastrowSymm") {
      m_ = netket::make_unique<JastrowSymm<T>>(graph, hilbert, pars);
    }
  }

  // TODO reove
  json ParsConv(const json &pars) {
    CheckFieldExists(pars, "Machine");
    return pars["Machine"];
  }

  template <class Partype>
  void InitParameters(const Partype &pars) {
    if (FieldOrDefaultVal(pars, "InitRandom", true)) {
      double sigma_rand = FieldOrDefaultVal(pars, "SigmaRand", 0.1);
      m_->InitRandomPars(1232, sigma_rand);

      InfoMessage() << "Machine initialized with random parameters"
                    << std::endl;
    }

    if (FieldExists(pars, "InitFile")) {
      std::string filename = pars["InitFile"];

      std::ifstream ifs(filename);

      if (ifs.is_open()) {
        json jmachine;
        ifs >> jmachine;
        m_->from_json(jmachine["Machine"]);
      } else {
        std::stringstream s;
        s << "Error opening file: " << filename;
        throw InvalidInputError(s.str());
      }

      InfoMessage() << "Machine initialized from file: " << filename
                    << std::endl;
    }
  }

  void CheckInput(const json &pars) {
    const std::string name = FieldVal<std::string>(pars, "Name", "Machine");

    std::set<std::string> machines = {
        "RbmSpin", "RbmSpinSymm", "RbmMultival", "FFNN",
        "Jastrow", "JastrowSymm", "MPSperiodic", "MPSdiagonal"};

    if (machines.count(name) == 0) {
      std::stringstream s;
      s << "Unknown Machine: " << name;
      throw InvalidInputError(s.str());
    }
  }

  // returns the number of variational parameters
  int Npar() const override { return m_->Npar(); }

  int Nvisible() const override { return m_->Nvisible(); }

  // Initializes Lookup tables
  void InitLookup(const Eigen::VectorXd &v, LookupType &lt) override {
    return m_->InitLookup(v, lt);
  }

  // Updates Lookup tables
  void UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    return m_->UpdateLookup(v, tochange, newconf, lt);
  }

  VectorType DerLog(const Eigen::VectorXd &v) override { return m_->DerLog(v); }

  VectorType GetParameters() override { return m_->GetParameters(); }

  void SetParameters(const VectorType &pars) override {
    return m_->SetParameters(pars);
  }

  // Value of the logarithm of the wave-function
  T LogVal(const Eigen::VectorXd &v) override { return m_->LogVal(v); }

  // Value of the logarithm of the wave-function
  // using pre-computed look-up tables for efficiency
  T LogVal(const Eigen::VectorXd &v, const LookupType &lt) override {
    return m_->LogVal(v, lt);
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      const Eigen::VectorXd &v, const std::vector<std::vector<int>> &toflip,
      const std::vector<std::vector<double>> &newconf) override {
    return m_->LogValDiff(v, toflip, newconf);
  }

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped Version using pre-computed look-up tables for efficiency
  // on a small number of spin flips
  T LogValDiff(const Eigen::VectorXd &v, const std::vector<int> &toflip,
               const std::vector<double> &newconf,
               const LookupType &lt) override {
    return m_->LogValDiff(v, toflip, newconf, lt);
  }

  void InitRandomPars(int seed, double sigma) override {
    return m_->InitRandomPars(seed, sigma);
  }

  const AbstractHilbert &GetHilbert() const override { return hilbert_; }

  void to_json(json &j) const override { m_->to_json(j); }

  void from_json(const json &j) override { m_->from_json(j); }
};
}  // namespace netket
#endif
