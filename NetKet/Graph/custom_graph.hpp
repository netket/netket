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
 public:
  using AbstractGraph::ColorMap;
  using AbstractGraph::Edge;

 private:
  // List of edges
  std::vector<Edge> edges_;

  ColorMap eclist_;

  int nsites_;

  std::vector<std::vector<int>> automorphisms_;

 public:
  CustomGraph(std::vector<Edge> edges, ColorMap colors = ColorMap(),
              std::vector<std::vector<int>> automorphisms =
                  std::vector<std::vector<int>>())
      : edges_{std::move(edges)},
        eclist_{std::move(colors)},
        automorphisms_{std::move(automorphisms)} {
    nsites_ = CheckEdges();
    if (nsites_ == 0) {
      throw InvalidInputError{"Empty graphs are not supported."};
    }
    if (eclist_.empty() && !edges_.empty()) {
      for (auto const &edge : edges_) {
        eclist_.emplace(edge, 0);
      }
    } else {
      CheckEdgeColors();
    }
    if (!automorphisms_.empty()) {
      CheckAutomorph();
    } else {
      automorphisms_.resize(1);
      automorphisms_.front().resize(static_cast<std::size_t>(nsites_));
      std::iota(std::begin(automorphisms_.front()),
                std::end(automorphisms_.front()), 0);
    }
  }

  /// Checks that for each edge (i, j): 0 <= i <= j and returns max(j) + 1, i.e.
  /// the number of nodes
  int CheckEdges() const {
    if (edges_.empty()) {
      return 0;
    }
    int min = 0;
    int max = -1;
    for (auto const &edge : edges_) {
      if (edge[0] > edge[1]) {
        throw InvalidInputError{
            "For each edge i<->j, i must not be greater than j"};
      }
      if (edge[0] < min) min = edge[0];
      if (edge[1] > max) max = edge[1];
    }
    if (min < 0) {
      throw InvalidInputError{"Nodes act as indices and should be >=0"};
    }
    assert(max >= min && "Bug! Postcondition violated");
    return max + 1;
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

  int Nsites() const noexcept override { return nsites_; }

  int Size() const noexcept override { return nsites_; }

  std::vector<Edge> Edges() const noexcept override { return edges_; }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return detail::AdjacencyListFromEdges(Edges(), Nsites());
  }

  bool IsBipartite() const noexcept override {
    WarningMessage() << "Assuming the custom graph is not bipartite\n";
    return false;
  }

  bool IsConnected() const noexcept override { return ComputeConnected(); }

  // Returns map of the edge and its respective color
  const ColorMap &EdgeColors() const noexcept override { return eclist_; }

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
