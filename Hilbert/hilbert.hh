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

#ifndef NETKET_HILBERT_HH
#define NETKET_HILBERT_HH

#include "Parallel/parallel.hh"
#include "abstract_hilbert.hh"
#include "bosons.hh"
#include "custom_hilbert.hh"
#include "next_variation.hh"
#include "qubits.hh"
#include "spins.hh"
#include "Json/json.hh"
#include <memory>
#include <set>

namespace netket {

class Hilbert : public AbstractHilbert {

  std::shared_ptr<AbstractHilbert> h_;

public:
  explicit Hilbert() {}

  explicit Hilbert(const Hilbert &oh) { h_ = oh.h_; }

  explicit Hilbert(const json &pars) { Init(pars); }

  void Init(const json &pars) {
    CheckInput(pars);

    if (FieldExists(pars["Hilbert"], "Name")) {
      if (pars["Hilbert"]["Name"] == "Spin") {
        h_ = std::make_shared<Spin>(pars);
      } else if (pars["Hilbert"]["Name"] == "Boson") {
        h_ = std::make_shared<Boson>(pars);
      } else if (pars["Hilbert"]["Name"] == "Qubit") {
        h_ = std::make_shared<Qubit>(pars);
      }
    } else {
      h_ = std::make_shared<CustomHilbert>(pars);
    }
  }

  void CheckInput(const json &pars) {
    int mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    if (!FieldExists(pars, "Hilbert")) {
      if (!FieldExists(pars, "Hamiltonian")) {
        if (mynode == 0)
          std::cerr << "Not enough information to construct Hilbert space"
                    << std::endl;
        std::abort();
      } else {
        if (!FieldExists(pars["Hamiltonian"], "Name")) {
          if (mynode == 0)
            std::cerr << "Not enough information to construct Hilbert space"
                      << std::endl;
          std::abort();
        }
      }
    }

    if (FieldExists(pars["Hilbert"], "Name")) {
      std::set<std::string> hilberts = {"Spin", "Boson", "Qubit"};

      const auto name = pars["Hilbert"]["Name"];

      if (hilberts.count(name) == 0) {
        std::cerr << "Hilbert " << name << " not found." << std::endl;
        std::abort();
      }
    }
  }

  bool IsDiscrete() const { return h_->IsDiscrete(); }

  int LocalSize() const { return h_->LocalSize(); }

  int Size() const { return h_->Size(); }

  std::vector<double> LocalStates() const { return h_->LocalStates(); }

  void RandomVals(Eigen::VectorXd &state,
                  netket::default_random_engine &rgen) const {
    return h_->RandomVals(state, rgen);
  }

  void UpdateConf(Eigen::VectorXd &v, const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const {
    return h_->UpdateConf(v, tochange, newconf);
  }
};
} // namespace netket
#endif
