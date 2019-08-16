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

#ifndef NETKET_LATTICE_HPP
#define NETKET_LATTICE_HPP

#include "abstract_graph.hpp"

namespace netket {

class Lattice : public AbstractGraph {
  int ndim_;
  int nsites_;         // number of nodes in the graph
  int natoms_;         // number of atoms in the unit cell
  int nlatticesites_;  // number of lattice sites (one lattice site can contain
                       // natoms_)
  std::vector<int> extent_;
  std::vector<bool> pbc_;
  std::vector<std::vector<double>> basis_vectors_;
  std::vector<std::vector<double>>
      atoms_coord_;                     // atoms position in the unit cell
  std::vector<std::vector<double>> R_;  // atoms position in the bravais lattice
  std::vector<Edge> edges_;
  std::vector<std::vector<int>> symmetrytable_;
  ColorMap colors_;  ///< Edge to color mapping
  bool is_connected_;
  bool is_bipartite_;

 public:
  // Constructor
  Lattice(std::vector<std::vector<double>> basis_vector,
          std::vector<int> extent, std::vector<bool> pbc,
          std::vector<std::vector<double>> V_atoms);
  ~Lattice();

  // Get private members

  /**
  Member function returning the dimensionality of the graph.
  @return Dimensionality of the graph.
  */
  int Ndim() const noexcept;

  int Nsites() const noexcept override;
  int Size() const noexcept override;

  /**
  Member function returning the basis vectors that define the lattice (graph).
  @return Basis vectors that define the lattice (graph).
  */
  std::vector<std::vector<double>> BasisVectors() const;

  // Graph properties #1
  /**
   Member function returning the coordinates of the graph sites (nodes).
   @return Coordinates of the graph sites (nodes).
   */
  std::vector<std::vector<double>> Coordinates() const;

  std::vector<Edge> const &Edges() const noexcept override;
  std::vector<std::vector<int>> SymmetryTable() const override;
  std::vector<std::vector<int>> AdjacencyList() const override;
  const ColorMap &EdgeColors() const noexcept override;

  // Graph sites representations (site = k, vector = n_i, coord = coordinates)

  /**
  Member function returning the vector of integers corresponding to the site i,
  where i is an integer. The output vector indicates how many translations of
  the basis vectors have been performed while building the graph.
  @param i Integer label associated to a graph node.
   */
  std::vector<int> Site2Vector(int i) const;

  /**
  Member function returning the coordinates of the i-th atom in the site
  labelled by n.
  @param n Vector of integers associated to a graph node (see above)
  @param iatom Label indicating which atom in the unit cell is considered
   */
  std::vector<double> Vector2Coord(const std::vector<int> &n, int iatom) const;

  /**
  Member function returning the coordinates of the k-th lattice site (graph
  node).
  @param k Integer label associated to a graph node.
  */
  std::vector<double> Site2Coord(int k) const;

  /**
  Member function returning the integer label associated to a graph node,
  given its vectorial characterizaion.
  @param n Vector of integers associated to a graph node (see above)
   */
  int Vector2Site(const std::vector<int> &n) const;

  /**
  Member function returning the label indicating which atom in the unit cell is
  associated to the k-th graph node.
  @param k Integer label associated to a graph node.
   */
  int AtomLabel(int k) const;

  /**
  Member function returning the integer label of the NEAREST neighbours of the
  k-th site.
  @param k Integer label associated to a graph node.
  @param iatom Label indicating which atom in the unit cell is considered
   */
  std::vector<int> FindNeighbours(int k, int iatom) const;

 private:
  // Graph properties #2
  std::vector<Edge> BuildEdges() const;
  std::vector<std::vector<int>> BuildSymmetryTable() const;

  // Nearest Neighbours Utils
  /**
  Member function computing the vector representation of all the "possible"
  nearest neighbours of a generic lattice site (k). The NN candidates are the
  ones living in sites which are translated by +1,0,-1 basis_vectors from site
  k. Each vector can have +1/0/-1, so all the permutations have to be
  considered.
   */
  std::vector<std::vector<int>> PossibleLatticeNeighbours() const;

  /**
  Member function computing the distances between the "possible" nearest
  neighbours.
   */
  std::vector<double> NeighboursSquaredDistance(
      const std::vector<std::vector<int>> &neighbours_matrix, int iatom) const;

  /**
  Member function computing the vector representation of all the nearest
  neighbours of the i-th atom in a generic lattice site. Basically it applies
  the distance criterion to the "possible" nearest neighbours (see above).
   */
  std::vector<std::vector<int>> LatticeNeighbours(int iatom) const;

  // Generic Utils
  /**
  Member function returning the norm of its argument.
   */
  double GetNorm(const std::vector<double> &coord) const;

  /**
  Member function returning the squared distance between its arguments.
   */
  double GetSquaredDistance(const std::vector<double> &v1,
                            const std::vector<double> &v2) const;
};

}  // namespace netket

#endif
