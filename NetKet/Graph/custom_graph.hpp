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

#include <algorithm>
#include <cassert>
#include <sstream>
#include <vector>
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
                  std::vector<std::vector<int>>(),
              bool isbipartite = false)
      : edges_{std::move(edges)},
        eclist_{std::move(colors)},
        is_bipartite_{isbipartite},
        automorphisms_{std::move(automorphisms)} {
    n_sites_ = CheckEdges();
    if (n_sites_ == 0) {
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
      automorphisms_.front().resize(static_cast<std::size_t>(n_sites_));
      std::iota(std::begin(automorphisms_.front()),
                std::end(automorphisms_.front()), 0);
    }
    is_connected_ = ComputeConnected();
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

  void CheckAutomorph() const {
    auto const print_list = [](std::ostream &os,
                               std::vector<int> const &xs) -> std::ostream & {
      os << "[";
      if (xs.size() >= 1) {
        os << xs.front();
      }
      for (auto i = std::size_t{0}; i < xs.size(); ++i) {
        os << ", " << xs[i];
      }
      return os << "]";
    };

    std::vector<std::uint8_t> been_here(static_cast<std::size_t>(n_sites_));

    auto const check_one = [&been_here,
                            print_list](std::vector<int> const &xs) {
      using std::begin;
      using std::end;
      std::fill(begin(been_here), end(been_here), std::uint8_t{0});
      // First, check that the cardinalities match
      if (xs.size() != been_here.size()) {
        std::ostringstream msg;
        msg << "Automorphism list is invalid: ";
        print_list(msg, xs);
        msg << " is not an automorphism: invalid dimension";
        throw InvalidInputError{msg.str()};
      }
      // Now, check that the i'th "automorphism" is surjective
      for (auto const x : xs) {
        if (x < 0 || x >= static_cast<int>(been_here.size())) {
          std::ostringstream msg;
          msg << "Automorphism list is invalid: ";
          print_list(msg, xs);
          msg << " is not an automorphism: " << x
              << " is not a valid site index";
          throw InvalidInputError{msg.str()};
        }
        been_here[static_cast<std::size_t>(x)] = 1;
      }
      if (!std::all_of(begin(been_here), end(been_here),
                       [](std::uint8_t x) { return x; })) {
        std::ostringstream msg;
        msg << "Automorphism list is invalid: ";
        print_list(msg, xs);
        msg << " is not an automorphism: is not a bijection";
        throw InvalidInputError{msg.str()};
      }
    };

    for (auto const &xs : automorphisms_) {
      check_one(xs);
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

  int Nsites() const noexcept override { return n_sites_; }

  int Size() const noexcept override { return n_sites_; }

  std::vector<Edge> const Edges() const noexcept override {
    return std::move(edges_);
  }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return detail::AdjacencyListFromEdges(Edges(), Nsites());
  }

  bool IsBipartite() const noexcept override { return is_bipartite_; }

  bool IsConnected() const noexcept override { return is_connected_; }

  // Returns map of the edge and its respective color
  const ColorMap EdgeColors() const noexcept override {
    return std::move(eclist_);
  }

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
