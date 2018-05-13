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

#ifndef NETKET_SAMPLER_HH
#define NETKET_SAMPLER_HH

#include "Parallel/parallel.hh"
#include "abstract_sampler.hh"
#include "metropolis_exchange.hh"
#include "metropolis_exchange_pt.hh"
#include "metropolis_hamiltonian.hh"
#include "metropolis_hamiltonian_pt.hh"
#include "metropolis_hop.hh"
#include "metropolis_local.hh"
#include "metropolis_local_pt.hh"
#include <memory>
#include <set>

namespace netket {

template <class WfType> class Sampler : public AbstractSampler<WfType> {

  using Ptype = std::unique_ptr<AbstractSampler<WfType>>;
  Ptype s_;

public:
  explicit Sampler(WfType &psi, const json &pars) {
    CheckInput(pars);
    Init(psi, pars);
  }

  explicit Sampler(Graph &graph, WfType &psi, const json &pars) {
    CheckInput(pars);
    Init(psi, pars);
    Init(graph, psi, pars);
  }

  explicit Sampler(Hamiltonian &hamiltonian, WfType &psi, const json &pars) {
    CheckInput(pars);
    Init(psi, pars);
    Init(hamiltonian, psi, pars);
  }

  explicit Sampler(Graph &graph, Hamiltonian &hamiltonian, WfType &psi,
                   const json &pars) {
    CheckInput(pars);
    Init(psi, pars);
    Init(graph, psi, pars);
    Init(hamiltonian, psi, pars);
  }

  void Init(WfType &psi, const json &pars) {
    if (pars["Sampler"]["Name"] == "MetropolisLocal") {
      s_ = Ptype(new MetropolisLocal<WfType>(psi));
    } else if (pars["Sampler"]["Name"] == "MetropolisLocalPt") {
      s_ = Ptype(new MetropolisLocalPt<WfType>(psi, pars));
    }
  }

  void Init(Graph &graph, WfType &psi, const json &pars) {
    if (pars["Sampler"]["Name"] == "MetropolisExchange") {
      s_ = Ptype(new MetropolisExchange<WfType>(graph, psi, pars));
    } else if (pars["Sampler"]["Name"] == "MetropolisExchangePt") {
      s_ = Ptype(new MetropolisExchangePt<WfType>(graph, psi, pars));
    } else if (pars["Sampler"]["Name"] == "MetropolisHop") {
      s_ = Ptype(new MetropolisHop<WfType>(graph, psi, pars));
    }
  }

  void Init(Hamiltonian &hamiltonian, WfType &psi, const json &pars) {
    if (pars["Sampler"]["Name"] == "MetropolisHamiltonian") {
      s_ = Ptype(
          new MetropolisHamiltonian<WfType, Hamiltonian>(psi, hamiltonian));
    } else if (pars["Sampler"]["Name"] == "MetropolisHamiltonianPt") {
      s_ = Ptype(new MetropolisHamiltonianPt<WfType, Hamiltonian>(
          psi, hamiltonian, pars));
    }
  }

  void CheckInput(const json &pars) {
    int mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    if (!FieldExists(pars, "Sampler")) {
      if (mynode == 0)
        std::cerr << "Sampler is not defined in the input" << std::endl;
      std::abort();
    }

    if (!FieldExists(pars["Sampler"], "Name")) {
      if (mynode == 0)
        std::cerr << "Sampler Name is not defined in the input" << std::endl;
      std::abort();
    }

    std::set<std::string> samplers = {
        "MetropolisLocal",       "MetropolisLocalPt",
        "MetropolisExchange",    "MetropolisExchangePt",
        "MetropolisHamiltonian", "MetropolisHamiltonianPt",
        "MetropolisHop"};

    const auto name = pars["Sampler"]["Name"];

    if (samplers.count(name) == 0) {
      std::cerr << "Sampler " << name << " not found." << std::endl;
      std::abort();
    }
  }

  void Reset(bool initrandom = false) { return s_->Reset(initrandom); }
  void Sweep() { return s_->Sweep(); }
  Eigen::VectorXd Visible() { return s_->Visible(); }
  void SetVisible(const Eigen::VectorXd &v) { return s_->SetVisible(v); }
  WfType &Psi() { return s_->Psi(); }
  Eigen::VectorXd Acceptance() const { return s_->Acceptance(); }
};
} // namespace netket

#endif
