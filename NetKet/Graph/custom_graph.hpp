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
#include <array>
#include <cassert>
#include <unordered_map>
#include <vector>
#include "Utils/all_utils.hpp"

namespace netket {

/**
    Class for user-defined graphs
    The list of edges and nodes is read from a json input file.
*/
class CustomGraph : public AbstractGraph {
  // adjacency list
  std::vector<std::vector<int>> adjlist_;

  ColorMap eclist_;

  int nsites_;

  std::vector<std::vector<int>> automorphisms_;

  bool isbipartite_;
  bool is_connected_;

 public:
  explicit CustomGraph(
      int size = 0,
      std::vector<std::vector<int>> adjl = std::vector<std::vector<int>>(),
      std::vector<std::vector<int>> edges = std::vector<std::vector<int>>(),
      std::vector<std::vector<int>> automorphisms =
          std::vector<std::vector<int>>(),
      std::vector<std::vector<int>> edgecolors =
          std::vector<std::vector<int>>(),
      bool isbipartite = false)
      : adjlist_(adjl), isbipartite_(isbipartite) {
    Init(size, edges, automorphisms, edgecolors);
  }

  void Init(int size, const std::vector<std::vector<int>> &edges,
            const std::vector<std::vector<int>> &automorphisms,
            const std::vector<std::vector<int>> &edgecolors) {
    bool has_edges = edges.size() > 0;
    if (has_edges) {
      AdjacencyListFromEdges(edges);
    }

    if (size != 0) {
      adjlist_.resize(size);
    }

    nsites_ = adjlist_.size();

    CheckGraph();

    is_connected_ = ComputeConnected();

    bool has_automorph = automorphisms.size() > 0;

    // Other graph properties
    if (has_automorph) {
      automorphisms_ = automorphisms;
    } else {
      automorphisms_.resize(1, std::vector<int>(nsites_));
      for (int i = 0; i < nsites_; i++) {
        // If no automorphism is specified, we stick to the identity one
        automorphisms_[0][i] = i;
      }
    }

    CheckAutomorph();

    bool has_edge_colors = edgecolors.size() > 0;

    if (has_edge_colors) {
      EdgeColorsFromList(edgecolors, eclist_);
    } else {
      EdgeColorsFromAdj(adjlist_, eclist_);
    }

    InfoMessage() << "Graph created " << std::endl;
    InfoMessage() << "Number of nodes = " << nsites_ << std::endl;
    if (has_automorph) {
      InfoMessage() << "Found automorphisms " << std::endl;
    }
    if (!has_edge_colors) {
      InfoMessage() << "No colors specified, edge colors set to 0 "
                    << std::endl;
    }
  }

  // TODO remove
  template <class Ptype>
  explicit CustomGraph(const Ptype &pars) {
    InitOld(pars);
  }

  // TODO remove
  template <class Ptype>
  void InitOld(const Ptype &pars) {
    // Try to construct from explicit graph definition

    if (FieldExists(pars, "AdjacencyList")) {
      std::vector<std::vector<int>> adjl =
          FieldVal<std::vector<std::vector<int>>>(pars, "AdjacencyList",
                                                  "Graph");
      adjlist_ = adjl;
    } else {
      adjlist_.resize(0);
    }

    if (FieldExists(pars, "Edges")) {
      std::vector<std::vector<int>> edges =
          FieldVal<std::vector<std::vector<int>>>(pars, "Edges", "Graph");
      AdjacencyListFromEdges(edges);
    }
    if (FieldExists(pars, "Size")) {
      int size = FieldVal<int>(pars, "Size");
      assert(size > 0);
      adjlist_.resize(size);
    }

    nsites_ = adjlist_.size();

    CheckGraph();

    is_connected_ = ComputeConnected();

    // Other graph properties
    if (FieldExists(pars, "Automorphisms")) {
      std::vector<std::vector<int>> ams =
          FieldVal<std::vector<std::vector<int>>>(pars, "Automorphisms",
                                                  "Graph");
      automorphisms_ = ams;
    } else {
      automorphisms_.resize(1, std::vector<int>(nsites_));
      for (int i = 0; i < nsites_; i++) {
        // If no automorphism is specified, we stick to the identity one
        automorphisms_[0][i] = i;
      }
    }

    CheckAutomorph();

    isbipartite_ = FieldOrDefaultVal<bool>(pars, "IsBipartite", false);

    // If edge colors are specificied read them in, otherwise set them all to
    // 0
    if (FieldExists(pars, "EdgeColors")) {
      std::vector<std::vector<int>> colorlist =
          FieldVal<std::vector<std::vector<int>>>(pars, "EdgeColors", "Graph");
      EdgeColorsFromList(colorlist, eclist_);
    } else {
      InfoMessage() << "No colors specified, edge colors set to 0 "
                    << std::endl;
      EdgeColorsFromAdj(adjlist_, eclist_);
    }

    InfoMessage() << "Graph created " << std::endl;
    InfoMessage() << "Number of nodes = " << nsites_ << std::endl;
  }

  void AdjacencyListFromEdges(const std::vector<std::vector<int>> &edges) {
    nsites_ = 0;

    for (auto edge : edges) {
      if (edge.size() != 2) {
        throw InvalidInputError(
            "The edge list is invalid (edges need "
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
    if (nsites_ < 1) {
      throw InvalidInputError("The number of graph nodes is invalid");
    }

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
  }

  void CheckAutomorph() {
    for (std::size_t i = 0; i < automorphisms_.size(); i++) {
      if (int(automorphisms_[i].size()) != nsites_) {
        throw InvalidInputError("The automorphism list is invalid");
      }
    }
  }
  void CheckEdgeColors() {
    // TODO write a meaningful check of edge colors
  }

  // Returns a list of permuted sites constituting an automorphism of the
  // graph
  std::vector<std::vector<int>> SymmetryTable() const override {
    return automorphisms_;
  }

  int Nsites() const override { return nsites_; }

  int Size() const override { return nsites_; }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return adjlist_;
  }

  bool IsBipartite() const override { return isbipartite_; }

  bool IsConnected() const override { return is_connected_; }

  // Returns map of the edge and its respective color
  const ColorMap &EdgeColors() const override { return eclist_; }

 private:
  bool ComputeConnected() const {
    const int start = 0;  // arbitrary node
    int nvisited = 0;
    BreadthFirstSearch(start, [&nvisited](int, int) { ++nvisited; });
    return nvisited == Nsites();
  }
};

}  // namespace netket
#endif
