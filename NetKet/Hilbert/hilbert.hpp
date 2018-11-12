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

#ifndef NETKET_HILBERT_HPP
#define NETKET_HILBERT_HPP

#include <memory>
#include <set>
#include "Graph/graph.hpp"
#include "Utils/json_utils.hpp"
#include "Utils/parallel_utils.hpp"
#include "abstract_hilbert.hpp"
#include "bosons.hpp"
#include "custom_hilbert.hpp"
#include "qubits.hpp"
#include "spins.hpp"

namespace netket {
// TODO remove
class Hilbert : public AbstractHilbert {
  std::unique_ptr<AbstractHilbert> h_;

 public:
  template <class Ptype>
  explicit Hilbert(const Ptype &pars) {
    InitWithoutGraph(ParsConv(pars));
  }

  template <class Ptype>
  explicit Hilbert(const Graph &graph, const Ptype &pars) {
    InitWithGraph(graph, ParsConv(pars));
  }

  json ParsConv(const json &pars) {
    CheckFieldExists(pars, "Hilbert");
    return pars["Hilbert"];
  }

  template <class Ptype>
  void InitWithoutGraph(const Ptype &pars) {
    int size = FieldVal<int>(pars, "Size");

    json gpars;
    gpars["Graph"]["Name"] = "Custom";
    gpars["Graph"]["Size"] = size;
    Graph graph(gpars);

    InitWithGraph(graph, pars);
  }

  template <class Ptype>
  void InitWithGraph(const Graph &graph, const Ptype &pars) {
    if (FieldExists(pars, "Name")) {
      std::string name = FieldVal<std::string>(pars, "Name");
      if (name == "Spin") {
        h_ = netket::make_unique<Spin>(graph, pars);
      } else if (name == "Boson") {
        h_ = netket::make_unique<Boson>(graph, pars);
      } else if (name == "Qubit") {
        h_ = netket::make_unique<Qubit>(graph, pars);
      } else if (name == "Custom") {
        h_ = netket::make_unique<CustomHilbert>(graph, pars);
      } else {
        std::stringstream s;
        s << "Unknown Hilbert type: " << name;
        throw InvalidInputError(s.str());
      }
    } else {
      h_ = netket::make_unique<CustomHilbert>(graph, pars);
    }
  }

  bool IsDiscrete() const override { return h_->IsDiscrete(); }

  int LocalSize() const override { return h_->LocalSize(); }

  int Size() const override { return h_->Size(); }

  std::vector<double> LocalStates() const override { return h_->LocalStates(); }

  void RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                  netket::default_random_engine &rgen) const override {
    return h_->RandomVals(state, rgen);
  }

  void UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                  const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const override {
    return h_->UpdateConf(v, tochange, newconf);
  }

  const AbstractGraph &GetGraph() const override { return h_->GetGraph(); }
};
}  // namespace netket

#endif
