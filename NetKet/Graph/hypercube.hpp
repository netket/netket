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

#ifndef NETKET_HYPERCUBE_HPP
#define NETKET_HYPERCUBE_HPP

#include "abstract_graph.hpp"

namespace netket {

class Hypercube : public AbstractGraph {
  int length_;               ///< Side length of the hypercube
  int n_dim_;                ///< Number of dimensions
  int n_sites_;              ///< Total number of nodes in the graph
  bool pbc_;                 ///< Whether to use periodic boundary conditions
  std::vector<Edge> edges_;  ///< List of graph edges
  std::vector<std::vector<int>>
      symm_table_;   ///< Vector of permutations (translations actually) under
                     ///  which the graph is invariant
  ColorMap colors_;  ///< Edge to color mapping

 public:
  Hypercube(Hypercube const &) = delete;
  Hypercube(Hypercube &&) noexcept = default;
  Hypercube &operator=(Hypercube const &) noexcept = delete;
  Hypercube &operator=(Hypercube &&) noexcept = default;

  Hypercube(int length, int n_dim = 1, bool pbc = true);

  // TODO(twesterhout): length is strictly speaking not needed, but then the
  // logic becomes too complicated for my taste :) Also, the performance of this
  // function will probably be pretty bad, by I don't think it matters much.
  Hypercube(int length, ColorMap colors);

  int Nsites() const noexcept override { return n_sites_; }

  int Size() const noexcept override { return n_sites_; }

  int Length() const noexcept { return length_; }

  int Ndim() const noexcept { return n_dim_; }

  const std::vector<Edge> &Edges() const noexcept override { return edges_; }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return detail::AdjacencyListFromEdges(Edges(), Nsites());
  }

  bool IsBipartite() const noexcept { return !pbc_ || length_ % 2 == 0; }

  bool IsConnected() const noexcept { return true; }

  // Returns map of the edge and its respective color
  const ColorMap &EdgeColors() const noexcept override { return colors_; }

  std::vector<std::vector<int>> SymmetryTable() const override {
    return symm_table_;
  }

 private:
  /// Given a site's coordinate as a `Ndim()`-dimensional vector, returns the
  /// site's index. The mapping is unique, but unspecified. I.e. the fact that
  /// sometimes `Coord2Site({1, 2, 0, 0, 1}) == X` and `Coord2Site({1, 2, 0, 0,
  /// 2}) == X+1` should not be relied on.
  int Coord2Site(std::vector<int> const &coord) const;

  /// Given a site's index, returns its coordinates as a `Ndim()`-dimensional
  /// vector.
  std::vector<int> Site2Coord(int site) const;

  // Given the length, ndim, and pbc, returns (nsites, edges).
  static std::tuple<int, std::vector<Edge>> BuildEdges(int length, int ndim,
                                                       bool pbc);

  static int Coord2Site(std::vector<int> const &coord, int length) noexcept;

  static std::vector<int> Site2Coord(int site, int length, int n_dim);

  static std::vector<std::vector<int>> BuildSymmTable(int length, int n_dim,
                                                      bool pbc, int n_sites);
};  // namespace netket

}  // namespace netket
#endif
