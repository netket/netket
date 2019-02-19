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

#include <algorithm>
#include <cassert>
#include <vector>
#include "Utils/next_variation.hpp"
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

  Hypercube(int const length, int const n_dim = 1, bool pbc = true)
      : length_{length}, n_dim_{n_dim}, pbc_{pbc}, edges_{}, symm_table_{} {
    if (length < 1) {
      throw InvalidInputError{
          "Side length of the hypercube must be at least 1"};
    } else if (length <= 2 && pbc) {
      throw InvalidInputError{
          "length<=2 hypercubes cannot have periodic boundary conditions"};
    }
    if (n_dim < 1) {
      throw InvalidInputError{"Hypercube dimension must be at least 1"};
    }

    std::tie(n_sites_, edges_) = BuildEdges(length, n_dim, pbc);
    symm_table_ = BuildSymmTable(length, n_dim, pbc, n_sites_);

    colors_.reserve(n_sites_);
    for (auto const &edge : edges_) {
      auto success = colors_.emplace(edge, 0).second;
      static_cast<void>(success);  // Make everyone happy in the NDEBUG case
      assert(success && "There should be no duplicate edges");
    }
  }

  // TODO(twesterhout): length is strictly speaking not needed, but then the
  // logic becomes too complicated for my taste :) Also, the performance of this
  // function will probably be pretty bad, by I don't think it matters much.
  Hypercube(int const length, ColorMap colors)
      : length_{length},
        n_dim_{},
        pbc_{},
        edges_{},
        symm_table_{},
        colors_{std::move(colors)} {
    using std::begin;
    using std::end;
    if (colors_.empty()) {
      throw InvalidInputError{
          "Side length of the hypercube must be at least 1"};
    }
    // First of all, we extract the list of edges
    edges_.reserve(colors_.size());
    std::transform(
        begin(colors_), end(colors_), std::back_inserter(edges_),
        [](std::pair<AbstractGraph::Edge, int> const &x) { return x.first; });
    // Verifies the edges and computes the number of sites
    n_sites_ = detail::CheckEdges(edges_);

    // We know that length_^ndim_ == nsites_, so we can solve it for ndim_
    auto count = length_;
    n_dim_ = 1;
    while (count < n_sites_) {
      ++n_dim_;
      count *= length_;
    }
    if (count != n_sites_) {
      throw InvalidInputError{
          "Specified length and color map are incompatible"};
    }

    // For periodic boundary conditions, there are exactly ndim_ * nsites_ edges
    // and for open bounrary conditions -- ndim_ * (nsites_ - nsites_/length).
    if (static_cast<std::size_t>(n_dim_ * n_sites_) == edges_.size()) {
      pbc_ = true;
    } else if (static_cast<std::size_t>(
                   n_dim_ * (n_sites_ - n_sites_ / length)) == edges_.size()) {
      pbc_ = false;
    } else {
      throw InvalidInputError{"Invalid color map"};
    }

    // Finally, we can check whether the edges we extracted from the color map
    // make any sense.
    auto const correct_edges = std::get<1>(BuildEdges(length_, n_dim_, pbc_));
    if (edges_.size() != correct_edges.size()) {
      throw InvalidInputError{"Invalid color map"};
    }
    for (auto const &edge : correct_edges) {
      // There is no anbiguity related to ordering: (i, j) vs (j, i), because
      // CheckEdges asserts that i <= j for each (i, j).
      if (!colors_.count(edge)) {
        throw InvalidInputError{"Invalid color map"};
      }
    }

    symm_table_ = BuildSymmTable(length_, n_dim_, pbc_, n_sites_);
  }

  int Nsites() const noexcept override { return n_sites_; }

  int Size() const noexcept override { return n_sites_; }

  int Length() const noexcept { return length_; }

  int Ndim() const noexcept { return n_dim_; }

  const std::vector<Edge> &Edges() const noexcept override { return edges_; }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return detail::AdjacencyListFromEdges(Edges(), Nsites());
  }

  bool IsBipartite() const noexcept override {
    return !pbc_ || length_ % 2 == 0;
  }

  bool IsConnected() const noexcept override { return true; }

  // Returns map of the edge and its respective color
  const ColorMap &EdgeColors() const noexcept override { return colors_; }

  std::vector<std::vector<int>> SymmetryTable() const override {
    return symm_table_;
  }

  /// Given a site's coordinate as a `Ndim()`-dimensional vector, returns the
  /// site's index. The mapping is unique, but unspecified. I.e. the fact that
  /// sometimes `Coord2Site({1, 2, 0, 0, 1}) == X` and `Coord2Site({1, 2, 0, 0,
  /// 2}) == X+1` should not be relied on.
  int Coord2Site(std::vector<int> const &coord) const {
    auto const print_coord = [&coord](std::ostream &os) -> std::ostream & {
      os << "[";
      if (coord.size() >= 1) {
        os << coord.front();
      }
      for (auto i = std::size_t{0}; i < coord.size(); ++i) {
        os << ", " << coord[i];
      }
      return os << "]";
    };
    auto const fail = [print_coord, this]() {
      std::ostringstream msg;
      msg << "Invalid coordinate ";
      print_coord(msg);
      msg << " for a " << n_dim_ << "-dimensional hypercube of side length "
          << length_;
      throw InvalidInputError{msg.str()};
    };

    if (coord.size() != static_cast<std::size_t>(n_dim_)) {
      fail();
    }
    // We need this loop, because Coord2Site is exposed to Python and it's
    // unacceptable to have Python interpreter terminating with a seg fault just
    // because of an invalid input.
    for (auto const x : coord) {
      if (x < 0 || x >= length_) {
        fail();
      }
    }
    return Coord2Site(coord, length_);
  }

  /// Given a site's index, returns its coordinates as a `Ndim()`-dimensional
  /// vector.
  std::vector<int> Site2Coord(int site) const {
    if (site < 0 || site >= Nsites()) {
      std::ostringstream msg;
      msg << "Invalid site index: `site` must be in [0, " << Nsites()
          << "), but got " << site;
      throw InvalidInputError{msg.str()};
    }
    return Site2Coord(site, length_, n_dim_);
  }

 private:
  // Given the length, ndim, and pbc, returns (nsites, edges).
  static std::tuple<int, std::vector<Edge>> BuildEdges(int const length,
                                                       int const ndim,
                                                       bool pbc) {
    assert(length >= (1 + static_cast<int>(pbc)) && ndim >= 1 &&
           "Bug! Must hold by construction");
    // NOTE: a double has 53 bits matissa, while an int only 32, so the
    // following should work fine if we assume that nsites_ fits into an int
    auto const _n_sites = std::pow(length, ndim);
    assert(_n_sites == std::trunc(_n_sites) && "Bug! Inexact arithmetic");
    auto const n_sites = static_cast<int>(_n_sites);

    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>(n_sites));
    std::vector<int> coord(ndim, 0);
    auto const max_pos = length - 1;
    auto site = 0;
    do {
      for (auto i = 1, dim = 0; dim < ndim; ++dim, i *= length) {
        // NOTE(twesterhout): Hoping that any normal optimising compiler will
        // move the if (pbc_) dispath to outside the while loop...
        if (pbc) {
          if (coord[dim] == max_pos) {
            edges.push_back({site - i * (length - 1), site});
          } else {
            edges.push_back({site, site + i});
          }
        } else {
          if (coord[dim] != max_pos) {
            edges.push_back({site, site + i});
          }
        }
      }
      ++site;
    } while (next_variation(coord.rbegin(), coord.rend(), max_pos));
    assert(site == n_sites && "Bug! Postcondition violated");
    return std::tuple<int, std::vector<Edge>>{n_sites, std::move(edges)};
  }

  static int Coord2Site(std::vector<int> const &coord,
                        int const length) noexcept {
    // NOTE(twesterhout): This is unsafe w.r.t. signed integer overflow. It's
    // highly unlikely that such big graphs will ever be used with NetKet, but
    // still.
    assert(length > 0);
    auto site = 0;
    auto scale = 1;
    for (auto const i : coord) {
      site += scale * i;
      scale *= length;
    }
    return site;
  }

  static std::vector<int> Site2Coord(int const site, int const length,
                                     int const n_dim) {
    assert(length > 0 && n_dim > 0 && site >= 0);
    std::vector<int> coord(n_dim);
    auto s = site;
    for (int i = 0; i < n_dim; ++i) {
      coord[i] = s % length;
      s /= length;
    }
    return coord;
  }

  static std::vector<std::vector<int>> BuildSymmTable(int const length,
                                                      int const n_dim, bool pbc,
                                                      int const n_sites) {
    if (!pbc) {
      // Identity automorphism
      std::vector<int> v(n_sites);
      std::iota(std::begin(v), std::end(v), 0);
      return std::vector<std::vector<int>>(1, v);
    };

    // contains sites coordinates
    std::vector<std::vector<int>> sites;
    for (auto i = 0; i < n_sites; ++i) {
      sites.push_back(Site2Coord(i, length, n_dim));
    }

    std::vector<std::vector<int>> permtable;
    permtable.reserve(n_sites);

    std::vector<int> transl_sites(n_sites);
    std::vector<int> ts(n_dim);

    for (int i = 0; i < n_sites; i++) {
      for (int p = 0; p < n_sites; p++) {
        for (int d = 0; d < n_dim; d++) {
          ts[d] = (sites[i][d] + sites[p][d]) % length;
        }
        transl_sites[p] = Coord2Site(ts, length);
      }
      permtable.push_back(transl_sites);
    }
    return permtable;
  }
};

}  // namespace netket
#endif
