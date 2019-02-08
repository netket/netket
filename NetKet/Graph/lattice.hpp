#ifndef NETKET_LATTICE_HPP
#define NETKET_LATTICE_HPP

#include <array>
#include <vector>
#include "abstract_graph.hpp"

namespace netket {

class Lattice : public AbstractGraph {
  int ndim_;
  int nlatticesites_;
  int nsites_;
  int natoms_;
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

  int Ndim() const noexcept;
  int Nsites() const noexcept override;
  int Size() const noexcept override;
  std::vector<std::vector<double>> Coordinates() const noexcept;
  std::vector<Edge> const &Edges() const noexcept override;
  std::vector<std::vector<int>> SymmetryTable() const noexcept override;
  std::vector<std::vector<int>> AdjacencyList() const noexcept override;
  const ColorMap &EdgeColors() const noexcept override;
  bool IsBipartite() const noexcept override;
  bool IsConnected() const noexcept override;

 private:
  // Lattice sites representations
  std::vector<int> Site2Vector(int i) const;
  std::vector<double> Vector2Coord(const std::vector<int> &n, int iatom) const;
  std::vector<double> Site2Coord(int k) const;
  int Vector2Site(const std::vector<int> &n) const;
  // Neighbours
  std::vector<std::vector<int>> PossibleLatticeNeighbours() const;
  std::vector<double> NeighboursSquaredDistance(
      const std::vector<std::vector<int>> &neighbours_matrix, int iatom) const;
  std::vector<std::vector<int>> LatticeNeighbours(int iatom) const;

  std::vector<int> FindNeighbours(int k, int iatom) const;
  // FindNeighbours returns the index of the neighbours of site k (supposing
  // that the site 0 is at the origin of the coordinates)

  // Edges and AdjacencyList
  std::vector<Edge> BuildEdges() const;
  // BuildEdges build the edges

  // Symmetries
  std::vector<std::vector<int>> BuildSymmetryTable() const;
  // BuildSymmetryTable build the symmetrytable = st[i][j] contiene la
  // i-esima permutazione equivalente dei siti

  double GetNorm(const std::vector<double> &coord) const;
  // GetNorm returns the norm (squared distance from 0) of its argument
  double GetSquaredDistance(const std::vector<double> &v1,
                            const std::vector<double> &v2) const;
  // GetSquaredDistance returns the distance between its argument
  bool RelativelyEqual(double a, double b,
                       double maxRelativeDiff = 1.0e-6) const;
  // RelativelyEqual gives true if a and b are equal up to an epsilon
  // (double comparison issue)

  bool ComputeConnected() const;
  bool ComputeBipartite() const;
};
}  // namespace netket

#include "lattice_impl.hpp"

#endif
