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

#include <mpi.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <map>
#include <unordered_map>
#include <vector>
#include "Utils/json_utils.hpp"
#include "Utils/next_variation.hpp"

namespace netket {

class Hypercube : public AbstractGraph {
  int L_;                    //< Side length of the hypercube
  int ndim_;                 //< Number of dimensions
  int nsites_;               //< Total number of nodes in the graph
  bool pbc_;                 //< Whether to use periodic boundary conditions
  std::vector<Edge> edges_;  //< List of graph edges
  std::vector<std::vector<int>>
      symm_table_;   //< Vector of permutations (translations actually) under
                     //  which the graph is invariant
  ColorMap colors_;  //< Edge to color mapping

#if 1  // TODO(twesterhout): This is to be removed
  // contains sites coordinates
  std::vector<std::vector<int>> sites_;

  // maps coordinates to site number
  std::map<std::vector<int>, int> coord2sites_;

  // adjacency list
  std::vector<std::vector<int>> adjlist_;

  // Edge colors
  ColorMap eclist_;
#endif

 public:
  Hypercube(int const L, int const ndim = 1, bool pbc = true)
      : L_{L}, ndim_{ndim}, pbc_{pbc}, edges_{}, symm_table_{} {
    if (L < 1) {
      throw InvalidInputError{
          "Side length of the hypercube must be at least 1"};
    } else if (L <= 2 && pbc) {
      throw InvalidInputError{
          "L<=2 hypercubes cannot have periodic boundary conditions"};
    }
    if (ndim < 1) {
      throw InvalidInputError{"Hypercube dimension must be at least 1"};
    }

    std::tie(nsites_, edges_) = BuildEdges(L, ndim, pbc);
    symm_table_ = BuildSymmTable(L, ndim, pbc, nsites_);

    colors_.reserve(nsites_);
    for (auto const &edge : edges_) {
      colors_.emplace(edge, 0);
    }
  }

  // TODO(twesterhout): L is strictly speaking not needed, but then the logic
  // becomes too complicated for my taste :)
  // Also, the performance of this function will probably be pretty bad, by I
  // don't think it matters much.
  Hypercube(int const L, ColorMap colors)
      : L_{L},
        ndim_{},
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
    nsites_ = detail::CheckEdges(edges_);

    // We know that L_^ndim_ == nsites_, so we can solve it for ndim_
    auto count = L_;
    ndim_ = 1;
    while (count < nsites_) {
      ++ndim_;
      count *= L_;
    }
    if (count != nsites_) {
      throw InvalidInputError{"Specified L and color map are incompatible"};
    }

    // For periodic boundary conditions, there are exactly ndim_ * nsites_ edges
    // and for open bounrary conditions -- ndim_ * (nsites_ - nsites_/L).
    if (static_cast<std::size_t>(ndim_ * nsites_) == edges_.size()) {
      pbc_ = true;
    } else if (static_cast<std::size_t>(ndim_ * (nsites_ - nsites_ / L)) ==
               edges_.size()) {
      pbc_ = false;
    } else {
      throw InvalidInputError{"Invalid color map"};
    }

    // Finally, we can check whether the edges we extracted from the color map
    // make any sense.
    auto const correct_edges = std::get<1>(BuildEdges(L_, ndim_, pbc_));
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

    symm_table_ = BuildSymmTable(L_, ndim_, pbc_, nsites_);
  }

  int Nsites() const override { return nsites_; }

  int Size() const override { return nsites_; }

  int Length() const { return L_; }

  int Ndim() const { return ndim_; }

  const std::vector<Edge> &Edges() const noexcept { return edges_; }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return detail::AdjacencyListFromEdges(Edges(), Nsites());
  }

  bool IsBipartite() const override { return !pbc_ || L_ % 2 == 0; }

  bool IsConnected() const override { return true; }

  // Returns map of the edge and its respective color
  const ColorMap &EdgeColors() const override { return colors_; }

  std::vector<std::vector<int>> SymmetryTable() const override {
    return symm_table_;
  }

 private:
  // Given the L, ndim, and pbc, returns (nsites, edges).
  static std::tuple<int, std::vector<Edge>> BuildEdges(int const L,
                                                       int const ndim,
                                                       bool pbc) {
    assert(L >= (1 + static_cast<int>(pbc)) && ndim >= 1 &&
           "Bug! Must hold by construction");
    // NOTE: a double has 53 bits matissa, while an int only 32, so the
    // following should work fine if we assume that nsites_ fits into an int
    auto const _n_sites = std::pow(L, ndim);
    assert(_n_sites == std::trunc(_n_sites) && "Bug! Inexact arithmetic");
    auto const n_sites = static_cast<int>(_n_sites);

    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>(n_sites));
    std::vector<int> coord(ndim, 0);
    auto const max_pos = L - 1;
    auto site = 0;
    do {
      for (auto i = 1, dim = 0; dim < ndim; ++dim, i *= L) {
        // NOTE(twesterhout): Hoping that any normal optimising compiler will
        // move the if (pbc_) dispath to outside the while loop...
        if (pbc) {
          if (coord[dim] == max_pos) {
            edges.push_back({site - i * (L - 1), site});
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

  static int Coord2Site(std::vector<int> const &coord, int const L) noexcept {
    assert(L >= 0);
    auto site = 0;
    auto scale = 1;
    for (auto const i : coord) {
      site += scale * i;
      scale *= L;
    }
    return site;
  }

  static std::vector<std::vector<int>> BuildSymmTable(int const L,
                                                      int const n_dim, bool pbc,
                                                      int const n_sites) {
    if (!pbc) {
      // Identity automorphism
      std::vector<int> v(n_sites);
      std::iota(std::begin(v), std::end(v), 0);
      return std::vector<std::vector<int>>(1, v);
    };

    // maps coordinates to site number
    std::map<std::vector<int>, int> coord2sites;

    // contains sites coordinates
    std::vector<std::vector<int>> sites;

    int ns = 0;
    std::vector<int> coord(n_dim, 0);
    do {
      sites.push_back(coord);
      coord2sites[coord] = ns;
      ns++;
    } while (netket::next_variation(coord.begin(), coord.end(), L - 1));

    std::vector<std::vector<int>> permtable;
    permtable.reserve(n_sites);

    std::vector<int> transl_sites(n_sites);
    std::vector<int> ts(n_dim);

    for (int i = 0; i < n_sites; i++) {
      for (int p = 0; p < n_sites; p++) {
        for (int d = 0; d < n_dim; d++) {
          ts[d] = (sites[i][d] + sites[p][d]) % L;
        }
        transl_sites[p] = coord2sites.at(ts);
      }
      permtable.push_back(transl_sites);
    }
    return permtable;
  }

  void CheckEdgeColors() {
    // TODO write a meaningful check of edge colors
  }

 public:
  // Everything below this point is to be removed and is left here for now for
  // json stuff to compile.

#if 0
  explicit Hypercube(int L, int ndim, bool pbc = true,
                     std::vector<std::vector<int>> edgecolors =
                         std::vector<std::vector<int>>())
      : L_(L), ndim_(ndim), pbc_(pbc) {
    Init(edgecolors);
  }
#endif

  void Init(const std::vector<std::vector<int>> &edgecolors) {
    assert(L_ > 0);
    assert(ndim_ >= 1);
    GenerateLatticePoints();
    GenerateAdjacencyList();

    bool has_edge_colors = edgecolors.size() > 0;

    if (has_edge_colors) {
      EdgeColorsFromList(edgecolors, eclist_);
    } else {
      EdgeColorsFromAdj(adjlist_, eclist_);
    }

    CheckEdgeColors();

    InfoMessage() << "Hypercube created " << std::endl;
    InfoMessage() << "Dimension = " << ndim_ << std::endl;
    InfoMessage() << "L = " << L_ << std::endl;
    InfoMessage() << "Pbc = " << pbc_ << std::endl;
    if (!has_edge_colors)
      InfoMessage() << "No colors specified, edge colors set to 0 "
                    << std::endl;
  }

  // TODO REMOVE
  template <class Ptype>
  explicit Hypercube(const Ptype &pars)
      : L_(FieldVal<int>(pars, "L", "Graph")),
        ndim_(FieldVal<int>(pars, "Dimension", "Graph")),
        pbc_(FieldOrDefaultVal(pars, "Pbc", true)) {
    if (pbc_ && L_ <= 2) {
      throw InvalidInputError(
          "L<=2 hypercubes cannot have periodic boundary conditions");
    }
    InitOld(pars);
  }

  // TODO REMOVE
  template <class Ptype>
  void InitOld(const Ptype &pars) {
    assert(L_ > 0);
    assert(ndim_ >= 1);
    GenerateLatticePoints();
    GenerateAdjacencyList();

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

    InfoMessage() << "Hypercube created " << std::endl;
    InfoMessage() << "Dimension = " << ndim_ << std::endl;
    InfoMessage() << "L = " << L_ << std::endl;
    InfoMessage() << "Pbc = " << pbc_ << std::endl;
  }

  void GenerateLatticePoints() {
    std::vector<int> coord(ndim_, 0);

    nsites_ = 0;
    do {
      sites_.push_back(coord);
      coord2sites_[coord] = nsites_;
      nsites_++;
    } while (netket::next_variation(coord.begin(), coord.end(), L_ - 1));
  }

  void GenerateAdjacencyList() {
    adjlist_.resize(nsites_);

    for (int i = 0; i < nsites_; i++) {
      std::vector<int> neigh(ndim_);
      std::vector<int> neigh2(ndim_);

      neigh = sites_[i];
      neigh2 = sites_[i];
      for (int d = 0; d < ndim_; d++) {
        if (pbc_) {
          neigh[d] = (sites_[i][d] + 1) % L_;
          neigh2[d] = ((sites_[i][d] - 1) % L_ + L_) % L_;
          int neigh_site = coord2sites_.at(neigh);
          int neigh_site2 = coord2sites_.at(neigh2);
          adjlist_[i].push_back(neigh_site);
          adjlist_[i].push_back(neigh_site2);
        } else {
          if ((sites_[i][d] + 1) < L_) {
            neigh[d] = (sites_[i][d] + 1);
            int neigh_site = coord2sites_.at(neigh);
            adjlist_[i].push_back(neigh_site);
            adjlist_[neigh_site].push_back(i);
          }
        }

        neigh[d] = sites_[i][d];
        neigh2[d] = sites_[i][d];
      }
    }
  }

#if 0
  // Returns a list of permuted sites equivalent with respect to
  // translation symmetry
  std::vector<std::vector<int>> SymmetryTable() const override {
    if (!pbc_) {
      throw InvalidInputError(
          "Cannot generate translation symmetries "
          "in the hypercube without PBC");
    }

    std::vector<std::vector<int>> permtable;

    std::vector<int> transl_sites(nsites_);
    std::vector<int> ts(ndim_);

    for (int i = 0; i < nsites_; i++) {
      for (int p = 0; p < nsites_; p++) {
        for (int d = 0; d < ndim_; d++) {
          ts[d] = (sites_[i][d] + sites_[p][d]) % L_;
        }
        transl_sites[p] = coord2sites_.at(ts);
      }
      permtable.push_back(transl_sites);
    }
    return permtable;
  }

 public:
  int Nsites() const override { return nsites_; }

  int Size() const override { return nsites_; }

  int Length() const { return L_; }

  int Ndim() const { return ndim_; }

  std::vector<std::vector<int>> Sites() const { return sites_; }

  std::vector<int> SiteCoord(int i) const { return sites_[i]; }

  const std::vector<Edge> &Edges() const noexcept { return edges_; }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return adjlist_;
  }

  std::map<std::vector<int>, int> Coord2Site() const { return coord2sites_; }

  int Coord2Site(const std::vector<int> &coord) const {
    return coord2sites_.at(coord);
  }

  // Returns map of the edge and its respective color
  const ColorMap &EdgeColors() const override { return colors_; }
#endif
};

}  // namespace netket
#endif
