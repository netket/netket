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

#include "lattice.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include "Utils/array_search.hpp"
#include "Utils/math_helpers.hpp"
#include "Utils/messages.hpp"
#include "Utils/next_variation.hpp"

namespace netket {
// Constructor
Lattice::Lattice(std::vector<std::vector<double>> basis_vector,
                 std::vector<int> extent, std::vector<bool> pbc,
                 std::vector<std::vector<double>> V_atoms)
    : extent_(std::move(extent)),
      pbc_(std::move(pbc)),
      basis_vectors_(std::move(basis_vector)),
      atoms_coord_(std::move(V_atoms)) {
  ndim_ = basis_vectors_.size();

  // Dimension consistency check #1
  for (int i = 0; i < ndim_; i++) {
    if (basis_vectors_[i].size() != static_cast<std::size_t>(ndim_)) {
      throw InvalidInputError{
          "Each element of basis_vectors must have ndim components.\n"};
    }
  }
  // Dimension consistency check #2
  if (extent_.size() != static_cast<std::size_t>(ndim_)) {
    throw InvalidInputError{"Extent must have ndim components.\n"};
  }

  // Default case: a single atom in the unit cell, located at the origin
  if (atoms_coord_.size() == 0) {
    atoms_coord_.resize(1, std::vector<double>(ndim_, 0));
  }

  natoms_ = atoms_coord_.size();

  // Dimension consistency check #3
  for (int i = 0; i < natoms_; i++) {
    if (atoms_coord_[i].size() != static_cast<std::size_t>(ndim_)) {
      throw InvalidInputError{
          "Each element of atoms_coord must have ndim components.\n"};
    }
  }

  // Default case for pbc is true
  if (pbc_.size() == 0) {
    pbc_.resize(ndim_, true);
  }

  // Dimension consistency check #4
  if (pbc_.size() != static_cast<std::size_t>(ndim_)) {
    throw InvalidInputError{"Pbc must have ndim components.\n"};
  }

  // PATHOLOGIC #1: non-positive extent and pbc in too small lattices)
  for (int i = 0; i < ndim_; i++) {
    if (extent_[i] <= 0) {
      throw InvalidInputError{"Extent components must be >0.\n"};
    }
    if (extent_[i] <= 2 and pbc_[i] == true) {
      throw InvalidInputError{"PBC are not allowed when extent<=2.\n"};
    }
  }

  nlatticesites_ = 1;
  for (int k = 0; k < ndim_; k++) {
    if (CheckProductOverflow(nlatticesites_, extent_[k]))
      throw InvalidInputError{"Extent product overflows!\n"};
    nlatticesites_ *= extent_[k];
  }

  // PATHOLOGIC #2: nlatticesites_=0,1
  if (nlatticesites_ <= 1) {
    throw InvalidInputError{
        "A well-defined lattice must have at least 2 sites.\n"};
  }

  nsites_ = nlatticesites_ * natoms_;

  // Build the Bravais lattice (coordinates)
  for (int k = 0; k < natoms_; k++) {
    for (int i = 0; i < nlatticesites_; i++) {
      std::vector<int> n = Site2Vector(i);
      std::vector<double> R =
          Vector2Coord(n, k);  // fill R with coord of atom k of site i
      R_.push_back(R);
    }
  }
  edges_ = BuildEdges();
  symmetrytable_ = BuildSymmetryTable();
  is_connected_ = IsConnected();
  is_bipartite_ = IsBipartite();

  colors_.reserve(nsites_);
  for (auto const &edge : edges_) {
    auto success = colors_.emplace(edge, 0).second;
    static_cast<void>(success);  // Make everyone happy in the NDEBUG case
    assert(success && "There should be no duplicate edges");
  }
}

Lattice::~Lattice(){}

// Get private members
int Lattice::Ndim() const noexcept { return ndim_; }
int Lattice::Nsites() const noexcept { return nsites_; }
int Lattice::Size() const noexcept { return Nsites(); }

std::vector<std::vector<double>> Lattice::BasisVectors() const {
  return basis_vectors_;
}

// Graph properties
std::vector<std::vector<double>> Lattice::Coordinates() const { return R_; }
std::vector<Lattice::Edge> const &Lattice::Edges() const noexcept {
  return edges_;
}
std::vector<std::vector<int>> Lattice::SymmetryTable() const {
  return symmetrytable_;
}
const AbstractGraph::ColorMap &Lattice::EdgeColors() const noexcept {
  return colors_;
}

// Graph sites representations (site = k, vector = n_i, coord = coordinates)
std::vector<int> Lattice::Site2Vector(int i) const {
  assert(i >= 0 && "Bug! Site index should be non-negative");
  int ndim = extent_.size();
  std::vector<int> result(ndim, 0);
  int ip;
  ip = i % nlatticesites_;
  int k = ndim - 1;
  while (ip > 0) {
    result[k] = ip % extent_[k];
    ip /= extent_[k];
    k--;
  }
  return result;
}

std::vector<double> Lattice::Vector2Coord(const std::vector<int> &n,
                                          int iatom) const {
  std::vector<double> R(ndim_, 0);
  assert(iatom >= 0 && "Bug! Atom index should be non-negative");
  for (int j = 0; j < ndim_; j++) {
    R[j] = atoms_coord_[iatom][j];
    for (int k = 0; k < ndim_; k++) R[j] += n[k] * basis_vectors_[k][j];
  }
  return R;
}

std::vector<double> Lattice::Site2Coord(int k) const {
  assert(k >= 0 && "Bug! Site index should be non-negative");
  return R_[k];
}

int Lattice::Vector2Site(const std::vector<int> &n) const {
  int k = 0;
  for (int i = 0; i < Ndim(); i++) {
    int base = 1;
    for (int j = i + 1; j < Ndim(); j++) {
      if (CheckProductOverflow(base, extent_[j]))
        throw InvalidInputError{"Extent product overflows!\n"};
      base *= extent_[j];
    }
    if (CheckProductOverflow(base, n[i]))
      throw InvalidInputError{"Extent product overflows!\n"};
    if (CheckSumOverflow(k, n[i] * base))
      throw InvalidInputError{"Sum overflow!\n"};
    k += n[i] * base;
  }
  return k;
}

int Lattice::AtomLabel(int k) const {
  assert(k >= 0 && "Bug! Site index should be non-negative");
  return k / nlatticesites_;
}

// Nearest Neighbours Utils
std::vector<std::vector<int>> Lattice::PossibleLatticeNeighbours() const {
  constexpr int nsymbols = 3;
  const int max = nsymbols - 1;
  std::vector<int> b(ndim_, 0);
  std::vector<int> result_vector;
  std::vector<std::vector<int>> result_matrix;
  std::vector<int> row(ndim_, 0);
  do {
    for (auto const &x : b) {
      result_vector.push_back(x);
    }
  } while (next_variation(b.begin(), b.end(), max));

  for (std::size_t i = 0; i < result_vector.size() / ndim_; i++) {
    for (int j = 0; j < ndim_; j++) {
      row[j] = result_vector[i * ndim_ + j] -
               1;  //-1 because I want (-1,0,1) as symbols instead of (0,1,2)
    }
    result_matrix.push_back(row);
  }
  return result_matrix;
}

std::vector<double> Lattice::NeighboursSquaredDistance(
    const std::vector<std::vector<int>> &neighbours_matrix, int iatom) const {
  std::vector<double> distance;
  assert(iatom >= 0 && "Bug! Atom index should be non-negative");
  for (std::size_t row = 0; row < neighbours_matrix.size(); row++) {
    std::vector<int> n(ndim_, 0);
    for (int j = 0; j < ndim_; j++) n[j] = neighbours_matrix[row][j];
    for (int jatom = 0; jatom < natoms_; jatom++)
      distance.push_back(
          GetSquaredDistance(Vector2Coord(n, jatom), atoms_coord_[iatom]));
  }

  return distance;
}

std::vector<std::vector<int>> Lattice::LatticeNeighbours(int iatom) const {
  std::vector<std::vector<int>> neighbours_matrix_in;
  std::vector<std::vector<int>> result;
  std::vector<double> distance;
  double min_distance;

  neighbours_matrix_in = PossibleLatticeNeighbours();

  distance = NeighboursSquaredDistance(neighbours_matrix_in, iatom);

  min_distance = *min_nonzero_elem(distance.begin(), distance.end());

  for (std::size_t i = 0; i < distance.size(); i++) {
    if (RelativelyEqual(distance[i], min_distance,
                        100. * std::numeric_limits<double>::epsilon())) {
      std::vector<int> temp;
      temp = neighbours_matrix_in[i / natoms_];
      temp.push_back(i % natoms_);
      result.push_back(temp);
    }
  }

  return result;
}

std::vector<int> Lattice::FindNeighbours(int k, int iatom) const {
  std::vector<int> result;
  std::vector<int> n = Site2Vector(k);
  std::vector<int> single_neighbour;

  std::vector<std::vector<int>> lattice_neighbours_vector =
      LatticeNeighbours(iatom);

  for (std::size_t i = 0; i < lattice_neighbours_vector.size(); i++) {
    std::vector<int> single_neighbour_vector;

    bool flag = true;
    for (int j = 0; j < Ndim(); j++) {
      int new_index = n[j] + lattice_neighbours_vector[i][j];
      if (pbc_[j]) {
        if (new_index < 0) {
          new_index = extent_[j] - 1;
        } else if (new_index >= extent_[j]) {
          new_index = 0;
        }
        single_neighbour_vector.push_back(new_index);

      } else if (new_index >= 0 and new_index < extent_[j]) {
        single_neighbour_vector.push_back(new_index);
      }

      else {
        flag = false;
      }  // if new_index refers to a border site, don't add this
         // site to the neighbours
    }
    if (flag) {
      single_neighbour.push_back(Vector2Site(single_neighbour_vector) +
                                 lattice_neighbours_vector[i][Ndim()] *
                                     nlatticesites_);
    }
  }

  return single_neighbour;
}

std::vector<Lattice::Edge> Lattice::BuildEdges() const {
  std::vector<Lattice::Edge> edge_vector;
  for (int k = 0; k < nlatticesites_; k++) {
    for (int iatom = 0; iatom < natoms_; iatom++) {
      std::vector<int> neighbours;

      neighbours = Lattice::FindNeighbours(k, iatom);

      for (std::size_t i = 0; i < neighbours.size(); i++) {
        if (k + iatom * nlatticesites_ < neighbours[i])
          edge_vector.push_back({{k + iatom * nlatticesites_, neighbours[i]}});
      }
    }
  }
  return edge_vector;
}

std::vector<std::vector<int>> Lattice::AdjacencyList() const {
  return detail::AdjacencyListFromEdges(edges_, nsites_);
}

std::vector<std::vector<int>> Lattice::BuildSymmetryTable() const {
  std::vector<std::vector<int>> integers_all;
  std::vector<std::vector<int>> result;
  for (int k = 0; k < nlatticesites_; k++)
    integers_all.push_back(Site2Vector(k));
  int i = 0;
  do {
    std::vector<int> row;
    int index = 1;
    for (int iatom = 0; iatom < natoms_; iatom++) {
      for (int j = 0; j < nlatticesites_; j++) {
        std::vector<int> n(ndim_);
        index = 1;
        for (int l = 0; l < ndim_; l++) {
          if (pbc_[l]) {
            n[l] = (integers_all[i][l] + integers_all[j][l]) % (extent_[l]);
          } else {
            n[l] = integers_all[j][l];
            index *= extent_[l];
          }
        }
        row.push_back(Vector2Site(n) + iatom * nlatticesites_);  // save rows
      }
    }
    i += index;
    result.push_back(row);
  } while (i < nlatticesites_);

  return result;
}

// Generic Utils

double Lattice::GetNorm(const std::vector<double> &coord) const {
  double distance = 0;
  for (std::size_t i = 0; i < coord.size(); i++)
    distance += coord[i] * coord[i];
  return distance;
}

double Lattice::GetSquaredDistance(const std::vector<double> &v1,
                                   const std::vector<double> &v2) const {
  double distance = 0;
  if (v1.size() != v2.size()) {
    throw InvalidInputError{
        "Impossible to compute distance between two vectors of "
        "different size.\n"};
  }
  for (std::size_t i = 0; i < v1.size(); i++)
    distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
  return distance;
}

}  // namespace netket
