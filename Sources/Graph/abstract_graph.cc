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

#include "abstract_graph.hpp"

namespace netket {

std::vector<std::vector<int>> AbstractGraph::AllDistances() const {
  // This can be implemented more efficiently (by reusing information between
  // calls to Distances) if necessary.
  std::vector<std::vector<int>> distances;
  for (int i = 0; i < Nsites(); i++) {
    distances.push_back(Distances(i));
  }
  return distances;
}

bool AbstractGraph::IsConnected() const noexcept {
  const int start = 0;  // arbitrary node
  int nvisited = 0;
  BreadthFirstSearch(start, [&nvisited](int, int) { ++nvisited; });
  return nvisited == Nsites();
}

bool AbstractGraph::IsBipartite() const noexcept {
  bool is_bipartite = true;
  const int start = 0;  // arbitrary node
  std::vector<int> colors(Nsites(), -1);
  const auto adjacency_list =
      AdjacencyList();  // implicit expression can't have
                        // access to the Lattice function
  if (IsConnected()) {
    BreadthFirstSearch(
        start, [&colors, &adjacency_list, &is_bipartite](int node, int) {
          if (node == start) colors[node] = 1;
          for (std::size_t j = 0; j < adjacency_list[node].size(); j++) {
            if (!is_bipartite) break;
            if (colors[adjacency_list[node][j]] == -1) {
              colors[adjacency_list[node][j]] = 1 - colors[node];
            } else if (colors[adjacency_list[node][j]] == colors[node]) {
              is_bipartite = false;
            }
          }
        });
  } else {
    BreadthFirstSearch(
        [&colors, &adjacency_list, &is_bipartite](int node, int, int) {
          if (node == start) colors[node] = 1;
          for (std::size_t j = 0; j < adjacency_list[node].size(); j++) {
            if (!is_bipartite) break;
            if (colors[adjacency_list[node][j]] == -1) {
              colors[adjacency_list[node][j]] = 1 - colors[node];
            } else if (colors[adjacency_list[node][j]] == colors[node]) {
              is_bipartite = false;
            }
          }
        });
  }
  return is_bipartite;
}

std::vector<int> AbstractGraph::Distances(int root) const {
  std::vector<int> dists(Nsites(), -1);

  // Dijkstra's algorithm
  BreadthFirstSearch(root,
                     [&dists](int node, int depth) { dists[node] = depth; });

  return dists;
}

void AbstractGraph::EdgeColorsFromList(
    const std::vector<std::vector<int>> &colorlist, ColorMap &eclist) {
  for (auto edge : colorlist) {
    eclist[{{edge[0], edge[1]}}] = edge[2];
  }
}

// If no Edge Colors are specified, initialize eclist_ with same color (0).
void AbstractGraph::EdgeColorsFromAdj(
    const std::vector<std::vector<int>> &adjlist, ColorMap &eclist) {
  for (int i = 0; i < static_cast<int>(adjlist.size()); i++) {
    for (std::size_t j = 0; j < adjlist[i].size(); j++) {
      eclist[{{i, adjlist[i][j]}}] = 0;
    }
  }
}

namespace detail {
/// Constructs the adjacency list given graph edges. No sanity checks are
/// performed. Use at your own risk!
std::vector<std::vector<int>> AdjacencyListFromEdges(
    const std::vector<AbstractGraph::Edge> &edges, int const number_sites) {
  assert(number_sites >= 0 && "Bug! Number of sites should be non-negative");
  std::vector<std::vector<int>> adjacency_list(
      static_cast<std::size_t>(number_sites));
  for (auto const &edge : edges) {
    adjacency_list[edge[0]].push_back(edge[1]);
    adjacency_list[edge[1]].push_back(edge[0]);
  }
  return adjacency_list;
}

int CheckEdges(std::vector<AbstractGraph::Edge> const &edges) {
  if (edges.empty()) {
    return 0;
  }
  int min = 0;
  int max = -1;
  for (auto const &edge : edges) {
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
}  // namespace detail

}  // namespace netket
