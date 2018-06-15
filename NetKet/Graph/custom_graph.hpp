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

#ifndef NETKET_CUSTOM_GRAPH_HPP
#define NETKET_CUSTOM_GRAPH_HPP

#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <vector>
#include "Hilbert/hilbert.hpp"
#include "Utils/json_utils.hpp"
#include "distance.hpp"

namespace netket {

/**
    Class for user-defined graphs
    The list of edges and nodes is read from a json input file.
*/
class CustomGraph : public AbstractGraph {
  // adjacency list
  std::vector<std::vector<int>> adjlist_;

  int nsites_;

  int mynode_;

  std::vector<std::vector<int>> automorphisms_;

  bool isbipartite_;

 public:
  // Json constructor
  explicit CustomGraph(const json &pars) { Init(pars); }

  void Init(const json &pars) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    // Try to construct from explicit graph definition
    if (FieldExists(pars, "Graph")) {
      if (FieldExists(pars["Graph"], "AdjacencyList")) {
        adjlist_ =
            pars["Graph"]["AdjacencyList"].get<std::vector<std::vector<int>>>();
      }
      if (FieldExists(pars["Graph"], "Edges")) {
        std::vector<std::vector<int>> edges =
            pars["Graph"]["Edges"].get<std::vector<std::vector<int>>>();
        AdjacencyListFromEdges(edges);
      }
      if (FieldExists(pars["Graph"], "Size")) {
        assert(pars["Graph"]["Size"] > 0);
        adjlist_.resize(pars["Graph"]["Size"]);
      }
    } else if (FieldExists(pars, "Hilbert")) {
      Hilbert hilbert(pars);
      nsites_ = hilbert.Size();
      assert(nsites_ > 0);
      adjlist_.resize(nsites_);
    } else {
      throw InvalidInputError("Graph: one among Size, AdjacencyList, Edges, or Hilbert "
                              "Space Size must be specified");
    }

    nsites_ = adjlist_.size();

    automorphisms_.resize(1, std::vector<int>(nsites_));
    for (int i = 0; i < nsites_; i++) {
      // If no automorphism is specified, we stick to the identity one
      automorphisms_[0][i] = i;
    }

    isbipartite_ = false;

    // Other graph properties
    if (FieldExists(pars, "Graph")) {
      if (FieldExists(pars["Graph"], "Automorphisms")) {
        automorphisms_ =
            pars["Graph"]["Automorphisms"].get<std::vector<std::vector<int>>>();
      }

      if (FieldExists(pars["Graph"], "IsBipartite")) {
        isbipartite_ = pars["Graph"]["IsBipartite"];
      }
    }

    CheckGraph();

    if (mynode_ == 0) {
      std::cout << "# Graph created " << std::endl;
      std::cout << "# Number of nodes = " << nsites_ << std::endl;
    }
  }

  void AdjacencyListFromEdges(const std::vector<std::vector<int>> &edges) {
    nsites_ = 0;

    for (auto edge : edges) {
      if (edge.size() != 2) {
        throw InvalidInputError("The edge list is invalid (edges need "
                                "to connect exactly two sites)");
      }
      if (edge[0] < 0 || edge[1] < 0) {
        throw InvalidInputError("The edge list is invalid");
      }

      nsites_ = std::max(std::max(edge[0], edge[1]), nsites_);
    }

    nsites_++;
    adjlist_.resize(nsites_);

    for (auto edge : edges) {
      adjlist_[edge[0]].push_back(edge[1]);
      adjlist_[edge[1]].push_back(edge[0]);
    }
  }

  void CheckGraph() {
    for (int i = 0; i < nsites_; i++) {
      for (auto s : adjlist_[i]) {
        // Checking if the referenced nodes are within the expected range
        if (s >= nsites_ || s < 0) {
          throw InvalidInputError("The graph is invalid");
        }
        // Checking if the adjacency list is symmetric
        // i.e. if site s is declared neihgbor of site i
        // when site i is declared neighbor of site s
        if (std::count(adjlist_[s].begin(), adjlist_[s].end(), i) != 1) {
          throw InvalidInputError("The graph adjacencylist is not symmetric");
        }
      }
    }
    for (std::size_t i = 0; i < automorphisms_.size(); i++) {
      if (int(automorphisms_[i].size()) != nsites_) {
        throw InvalidInputError("The automorphism list is invalid");
      }
    }
  }

  // Returns a list of permuted sites constituting an automorphism of the
  // graph
  std::vector<std::vector<int>> SymmetryTable() const { return automorphisms_; }

  int Nsites() const { return nsites_; }

  std::vector<std::vector<int>> AdjacencyList() const { return adjlist_; }

  bool IsBipartite() const { return isbipartite_; }

  // returns the distances of each point from the others
  std::vector<std::vector<int>> Distances() const {
    std::vector<std::vector<int>> distances;

    for (int i = 0; i < nsites_; i++) {
      distances.push_back(FindDist(adjlist_, i));
    }

    return distances;
  }
};

}  // namespace netket
#endif
