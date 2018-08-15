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
#include <cassert>
#include <iostream>
#include <map>
#include <vector>
#include "Utils/json_utils.hpp"
#include "distance.hpp"

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

  int nsites_;

 public:
  // Json constructor
  explicit Hypercube(const json &pars)
      : L_(FieldVal(pars["Graph"], "L", "Graph")),
        ndim_(FieldVal(pars["Graph"], "Dimension", "Graph")),
        pbc_(FieldOrDefaultVal(pars["Graph"], "Pbc", true)) {
    if (pbc_ && L_ <= 2) {
      throw InvalidInputError(
          "L<=2 hypercubes cannot have periodic boundary conditions");
    }
    Init();
  }

  void Init() {
    assert(L_ > 0);
    assert(ndim_ >= 1);
    GenerateLatticePoints();
    GenerateAdjacencyList();

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

      neigh = sites_[i];

      for (int d = 0; d < ndim_; d++) {
        if (pbc_) {
          neigh[d] = (sites_[i][d] + 1) % L_;
          int neigh_site = coord2sites_.at(neigh);
          adjlist_[i].push_back(neigh_site);
          adjlist_[neigh_site].push_back(i);
        } else {
          if ((sites_[i][d] + 1) < L_) {
            neigh[d] = (sites_[i][d] + 1);
            int neigh_site = coord2sites_.at(neigh);
            adjlist_[i].push_back(neigh_site);
            adjlist_[neigh_site].push_back(i);
          }
        }

        neigh[d] = sites_[i][d];
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

  // returns the distances of each point from the others
  std::vector<std::vector<int>> Distances() const override {
    std::vector<std::vector<int>> distances;

    for (int i = 0; i < nsites_; i++) {
      distances.push_back(FindDist(adjlist_, i));
    }

    return distances;
  }
};

}  // namespace netket
#endif
