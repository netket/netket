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

#ifndef NETKET_GRAPH_HH
#define NETKET_GRAPH_HH

#include <iostream>
#include <map>
#include <vector>

#include "abstract_graph.hh"
#include "custom_graph.hh"
#include "hypercube.hh"
#include "Json/json.hh"

namespace netket {

class Graph : public AbstractGraph {

  using Ptype = std::unique_ptr<AbstractGraph>;
  Ptype g_;

public:
  Graph(const json &pars) {

    // Check if a graph is explicitely defined in the input
    if (FieldExists(pars, "Graph")) {

      // Checking if we are using a graph in the hard-coded library
      if (FieldExists(pars["Graph"], "Name")) {
        if (pars["Graph"]["Name"] == "Hypercube") {
          g_ = Ptype(new Hypercube(pars));
        } else {
          std::cout << "Graph not found" << std::endl;
          std::abort();
        }
      }
      // Otherwise using a user-defined graph
      else {
        g_ = Ptype(new CustomGraph(pars));
      }
    } else {
      // Otherwise try to construct a custom graph using Hilbert space
      // information
      g_ = Ptype(new CustomGraph(pars));
    }
  }

  int Nsites() const { return g_->Nsites(); }
  std::vector<std::vector<int>> AdjacencyList() const {
    return g_->AdjacencyList();
  }
  std::vector<std::vector<int>> SymmetryTable() const {
    return g_->SymmetryTable();
  }
  std::vector<std::vector<int>> Distances() const { return g_->Distances(); }
  bool IsBipartite() const { return g_->IsBipartite(); }
};
} // namespace netket

#endif
