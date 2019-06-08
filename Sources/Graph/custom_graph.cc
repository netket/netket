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

#include "custom_graph.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace netket {

CustomGraph::CustomGraph(std::vector<Edge> edges, ColorMap colors,
                         std::vector<std::vector<int>> automorphisms)
    : edges_{std::move(edges)},
      eclist_{std::move(colors)},
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
  is_connected_ = IsConnected();
  is_bipartite_ = IsBipartite();
}

int CustomGraph::Nsites() const noexcept { return n_sites_; }

int CustomGraph::Size() const noexcept { return n_sites_; }

std::vector<CustomGraph::Edge> const &CustomGraph::Edges() const noexcept {
  return edges_;
}

std::vector<std::vector<int>> CustomGraph::AdjacencyList() const {
  return detail::AdjacencyListFromEdges(Edges(), Nsites());
}

const CustomGraph::ColorMap &CustomGraph::EdgeColors() const noexcept {
  return eclist_;
}

// Returns a list of permuted sites constituting an automorphism of the
// graph
std::vector<std::vector<int>> CustomGraph::SymmetryTable() const {
  return automorphisms_;
}

int CustomGraph::CheckEdges() const {
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

void CustomGraph::CheckAutomorph() const {
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

  auto const check_one = [&been_here, print_list](std::vector<int> const &xs) {
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
        msg << " is not an automorphism: " << x << " is not a valid site index";
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

void CustomGraph::CheckEdgeColors() const {
  // TODO write a meaningful check of edge colors
}

}  // namespace netket
