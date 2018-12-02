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

#ifndef NETKET_GRAPH_HPP
#define NETKET_GRAPH_HPP

#include <array>
#include <unordered_map>
#include <vector>
#include "Utils/json_utils.hpp"
#include "Utils/memory_utils.hpp"
#include "abstract_graph.hpp"
#include "custom_graph.hpp"
#include "hypercube.hpp"
#include "mpark/variant.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace netket {
class Graph : public AbstractGraph {
 public:
  using VariantType = mpark::variant<Hypercube, CustomGraph>;

 private:
  VariantType obj_;

 public:
  Graph(VariantType obj) : obj_(std::move(obj)) {}

  /**
  Member function returning the number of sites (nodes) in the graph.
  @return Number of sites (nodes) in the graph.
  */
  int Nsites() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.Nsites(); }, obj_);
  }

  /**
  Member function returning the number of sites (nodes) in the graph.
  @return Number of sites (nodes) in the graph.
  */
  int Size() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.Size(); }, obj_);
  }

  /**
  Returns the graph edges.
  */
  std::vector<Edge> const Edges() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.Edges(); }, obj_);
  }

  /**
  Member function returning the adjacency list of the graph.
  @return adl[i][k] is the k-th neighbour of site i.
  */
  std::vector<std::vector<int>> AdjacencyList() const override {
    return mpark::visit([](auto &&obj) { return obj.AdjacencyList(); }, obj_);
  }

  /**
  Member function returning the symmetry table of the graph.
  @return st[i][k] contains the i-th equivalent permutation of the sites.
  */
  std::vector<std::vector<int>> SymmetryTable() const override {
    return mpark::visit([](auto &&obj) { return obj.SymmetryTable(); }, obj_);
  }

  /**
  Member function returning edge colors of the graph.
  @return ec[i][j] is the color of the edge between nodes i and j.
  */
  const ColorMap EdgeColors() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.EdgeColors(); }, obj_);
  }

  /**
  Member function returning true if the graph is bipartite.
  @return true if lattice is bipartite.
  */
  bool IsBipartite() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.IsBipartite(); }, obj_);
  }

  /**
   * Checks whether the graph is connected, i.e., there exists a path between
   * every pair of nodes.
   * @return true, if the graph is connected
   */
  bool IsConnected() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.IsConnected(); }, obj_);
  }

  /**
   * Computes the distances of all nodes from a root node (single-source
   * shortest path problem). The distance of nodes not reachable from root
   are
   * set to -1.
   * @param root The root node from which the distances are calculated.
   * @return A vector `dist` of distances where dist[v] is the distance of v
   to
   * root.
   */
  std::vector<int> Distances(int root) const override {
    return mpark::visit([root](auto &&obj) { return obj.Distances(root); },
                        obj_);
  }

  /**
   * Computes the distances between each pair of nodes on the graph (the
   * all-pairs shortest path problem).
   * @return A vector of vectors dist where dist[v][w] is the distance
   between
   *    nodes v and w.
   */
  std::vector<std::vector<int>> AllDistances() const override {
    return mpark::visit([](auto &&obj) { return obj.AllDistances(); }, obj_);
  }
};
}  // namespace netket
#endif
