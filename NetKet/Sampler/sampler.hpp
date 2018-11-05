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

#ifndef NETKET_SAMPLER_HPP
#define NETKET_SAMPLER_HPP

#include <memory>
#include <set>
#include "Graph/graph.hpp"
#include "Hamiltonian/hamiltonian.hpp"
#include "Utils/memory_utils.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/python_helper.hpp"
#include "abstract_sampler.hpp"
#include "custom_sampler.hpp"
#include "custom_sampler_pt.hpp"
#include "exact_sampler.hpp"
#include "metropolis_exchange.hpp"
#include "metropolis_exchange_pt.hpp"
#include "metropolis_hamiltonian.hpp"
#include "metropolis_hamiltonian_pt.hpp"
#include "metropolis_hop.hpp"
#include "metropolis_local.hpp"
#include "metropolis_local_pt.hpp"

namespace netket {

template <class WfType>
class Sampler : public AbstractSampler<WfType> {
  std::unique_ptr<AbstractSampler<WfType>> s_;

 public:
  template <class Ptype>
  explicit Sampler(WfType &psi, const Ptype &pars) {
    const auto pconv = ParsConv(pars);
    CheckInput(pconv);
    Init(psi, pconv);
  }

  template <class Ptype>
  explicit Sampler(const Graph &graph, WfType &psi, const Ptype &pars) {
    const auto pconv = ParsConv(pars);
    CheckInput(pconv);
    Init(psi, pconv);
    Init(graph, psi, pconv);
  }

  template <class Ptype>
  explicit Sampler(Hamiltonian &hamiltonian, WfType &psi, const Ptype &pars) {
    const auto pconv = ParsConv(pars);
    CheckInput(pconv);
    Init(psi, pconv);
    Init(hamiltonian, psi, pconv);
  }

  template <class Ptype>
  explicit Sampler(const Graph &graph, Hamiltonian &hamiltonian, WfType &psi,
                   const Ptype &pars) {
    const auto pconv = ParsConv(pars);
    CheckInput(pconv);
    Init(psi, pconv);
    Init(graph, psi, pconv);
    Init(hamiltonian, psi, pconv);
  }

  template <class Ptype>
  void Init(WfType &psi, const Ptype &pars) {
    if (FieldExists(pars, "Name")) {
      std::string name = FieldVal<std::string>(pars, "Name");
      if (name == "MetropolisLocal") {
        s_ = netket::make_unique<MetropolisLocal<WfType>>(psi);
      } else if (name == "MetropolisLocalPt") {
        s_ = netket::make_unique<MetropolisLocalPt<WfType>>(psi, pars);
      } else if (name == "Exact") {
        s_ = netket::make_unique<ExactSampler<WfType>>(psi);
      }
    } else {
      if (FieldExists(pars, "Nreplicas")) {
        s_ = netket::make_unique<CustomSamplerPt<WfType>>(psi, pars);
      } else {
        s_ = netket::make_unique<CustomSampler<WfType>>(psi, pars);
      }
    }
  }

  template <class Ptype>
  void Init(const Graph &graph, WfType &psi, const Ptype &pars) {
    if (FieldExists(pars, "Name")) {
      std::string name = FieldVal<std::string>(pars, "Name");
      if (name == "MetropolisExchange") {
        s_ = netket::make_unique<MetropolisExchange<WfType>>(graph, psi, pars);
      } else if (name == "MetropolisExchangePt") {
        s_ =
            netket::make_unique<MetropolisExchangePt<WfType>>(graph, psi, pars);
      } else if (name == "MetropolisHop") {
        s_ = netket::make_unique<MetropolisHop<WfType>>(graph, psi, pars);
      }
    }
  }

  template <class Ptype>
  void Init(Hamiltonian &hamiltonian, WfType &psi, const Ptype &pars) {
    if (FieldExists(pars, "Name")) {
      std::string name = FieldVal<std::string>(pars, "Name");
      if (name == "MetropolisHamiltonian") {
        s_ = netket::make_unique<MetropolisHamiltonian<WfType, Hamiltonian>>(
            psi, hamiltonian);
      } else if (name == "MetropolisHamiltonianPt") {
        s_ = netket::make_unique<MetropolisHamiltonianPt<WfType, Hamiltonian>>(
            psi, hamiltonian, pars);
      }
    }
  }

  json ParsConv(const json &pars) {
    CheckFieldExists(pars, "Sampler");
    return pars["Sampler"];
  }
  pybind11::kwargs ParsConv(const pybind11::kwargs &pars) { return pars; }

  template <class Ptype>
  void CheckInput(const Ptype &pars) {
    int mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    if (FieldExists(pars, "Name")) {
      std::set<std::string> samplers = {
          "MetropolisLocal",       "MetropolisLocalPt",
          "MetropolisExchange",    "MetropolisExchangePt",
          "MetropolisHamiltonian", "MetropolisHamiltonianPt",
          "MetropolisHop",         "Exact",
      };

      std::string name = FieldVal<std::string>(pars, "Name");

      if (samplers.count(name) == 0) {
        std::stringstream s;
        s << "Unknown Sampler.Name: " << name;
        throw InvalidInputError(s.str());
      }
    } else {
      if (!FieldExists(pars, "ActingOn") and
          !FieldExists(pars, "MoveOperators")) {
        throw InvalidInputError(
            "No SamplerName provided or Custom Sampler (MoveOperators and "
            "ActingOn) defined");
      }
    }
  }

  void Reset(bool initrandom = false) override { return s_->Reset(initrandom); }

  void Sweep() override { return s_->Sweep(); }

  Eigen::VectorXd Visible() override { return s_->Visible(); }

  void SetVisible(const Eigen::VectorXd &v) override {
    return s_->SetVisible(v);
  }

  WfType &Psi() override { return s_->Psi(); }

  Eigen::VectorXd Acceptance() const override { return s_->Acceptance(); }
};
}  // namespace netket

#endif
