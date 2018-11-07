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
#include <array>
#include <cassert>
#include <map>
#include <unordered_map>
#include <vector>
#include "Utils/json_utils.hpp"
#include "Utils/next_variation.hpp"

namespace netket {

class Hypercube : public AbstractGraph {
  // edge of the hypercube
  const int L_;

  // number of dimensions
  const int ndim_;

  // whether to use periodic boundary conditions
  const bool pbc_;

  // contains sites coordinates
  std::vector<std::vector<int>> sites_;

  // maps coordinates to site number
  std::map<std::vector<int>, int> coord2sites_;

  // adjacency list
  std::vector<std::vector<int>> adjlist_;

  // Edge colors
  ColorMap eclist_;

  int nsites_;

 public:
  explicit Hypercube(int L, int ndim, bool pbc = true,
                     std::vector<std::vector<int>> edgecolors =
                         std::vector<std::vector<int>>())
      : L_(L), ndim_(ndim), pbc_(pbc) {
    Init(edgecolors);
  }

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

  void CheckEdgeColors() {
    // TODO write a meaningful check of edge colors
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

  int Nsites() const override { return nsites_; }

  int Size() const override { return nsites_; }

  int Length() const { return L_; }

  int Ndim() const { return ndim_; }

  std::vector<std::vector<int>> Sites() const { return sites_; }

  std::vector<int> SiteCoord(int i) const { return sites_[i]; }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return adjlist_;
  }

  std::map<std::vector<int>, int> Coord2Site() const { return coord2sites_; }

  int Coord2Site(const std::vector<int> &coord) const {
    return coord2sites_.at(coord);
  }

  bool IsBipartite() const override { return true; }

  bool IsConnected() const override { return true; }

  // Returns map of the edge and its respective color
  const ColorMap &EdgeColors() const override { return eclist_; }
};

}  // namespace netket
#endif
