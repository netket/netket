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

#include "abstract_graph.hpp"

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
  std::vector<Edge> edges_;  ///< List of graph edges
  ColorMap eclist_;          ///< Edge to color mapping
  int n_sites_;              ///< Total number of nodes in the graph
  bool is_connected_;        ///< Whether the graph is connected
  bool is_bipartite_;        ///< Whether the graph is bipartite
  std::vector<std::vector<int>> automorphisms_;

 public:
  CustomGraph(std::vector<Edge> edges, ColorMap colors = ColorMap(),
              std::vector<std::vector<int>> automorphisms =
                  std::vector<std::vector<int>>());

  int Nsites() const noexcept override;
  int Size() const noexcept override;
  std::vector<Edge> const &Edges() const noexcept override;
  std::vector<std::vector<int>> AdjacencyList() const override;
  const ColorMap &EdgeColors() const noexcept override;
  std::vector<std::vector<int>> SymmetryTable() const override;

 private:
  /// Checks that for each edge (i, j): 0 <= i <= j and returns max(j) + 1, i.e.
  /// the number of nodes
  int CheckEdges() const;
  void CheckAutomorph() const;
  void CheckEdgeColors() const;
};

}  // namespace netket

#endif  // NETKET_CUSTOM_GRAPH_HPP
