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

#include "abstract_machine.hpp"
#include "rbm_multival.hpp"
#include "rbm_spin.hpp"
#include "rbm_spin_symm.hpp"

namespace netket {

template <class T>
class Machine : public AbstractMachine<T> {
  using Ptype = std::unique_ptr<AbstractMachine<T>>;

  Ptype m_;

  const Hilbert &hilbert_;

 public:
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  explicit Machine(const Hilbert &hilbert, const json &pars)
      : hilbert_(hilbert) {
    CheckInput(pars);
    Init(hilbert_, pars);
    InitParameters(pars);
  }

  explicit Machine(const Hamiltonian &hamiltonian, const json &pars)
      : hilbert_(hamiltonian.GetHilbert()) {
    CheckInput(pars);
    Init(hilbert_, pars);
    InitParameters(pars);
  }

  explicit Machine(const Graph &graph, const Hilbert &hilbert, const json &pars)
      : hilbert_(hilbert) {
    CheckInput(pars);
    Init(hilbert_, pars);
    Init(graph, hilbert, pars);
    InitParameters(pars);
  }

  explicit Machine(const Graph &graph, const Hamiltonian &hamiltonian,
                   const json &pars)
      : hilbert_(hamiltonian.GetHilbert()) {
    CheckInput(pars);
    Init(hilbert_, pars);
    Init(graph, hilbert_, pars);
    InitParameters(pars);
  }

  void Init(const Hilbert &hilbert, const json &pars) {
    if (pars["Machine"]["Name"] == "RbmSpin") {
      m_ = Ptype(new RbmSpin<T>(hilbert, pars));
    } else if (pars["Machine"]["Name"] == "RbmMultival") {
      m_ = Ptype(new RbmMultival<T>(hilbert, pars));
    }
  }

  void Init(const Graph &graph, const Hilbert &hilbert, const json &pars) {
    if (pars["Machine"]["Name"] == "RbmSpinSymm") {
      m_ = Ptype(new RbmSpinSymm<T>(graph, hilbert, pars));
    }
  }

  void InitParameters(const json &pars) {
    if (FieldOrDefaultVal(pars["Machine"], "InitRandom", true)) {
      double sigma_rand = FieldOrDefaultVal(pars["Machine"], "SigmaRand", 0.1);
      m_->InitRandomPars(1232, sigma_rand);

      InfoMessage() << "Machine initialized with random parameters"
                    << std::endl;
    }

    if (FieldExists(pars["Machine"], "InitFile")) {
      std::string filename = pars["Machine"]["InitFile"];

      std::ifstream ifs(filename);

      if (ifs.is_open()) {
        json jmachine;
        ifs >> jmachine;
        m_->from_json(jmachine);
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
    CheckFieldExists(pars, "Machine");
    const std::string name = FieldVal(pars["Machine"], "Name", "Machine");

    std::set<std::string> machines = {"RbmSpin", "RbmSpinSymm", "RbmMultival"};

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

  const Hilbert &GetHilbert() const { return hilbert_; }

  void to_json(json &j) const override { m_->to_json(j); }

  void from_json(const json &j) override { m_->from_json(j); }
};
}  // namespace netket
#endif
