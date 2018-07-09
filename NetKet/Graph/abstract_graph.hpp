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

#ifndef NETKET_ABSTRACTGRAPH_HPP
#define NETKET_ABSTRACTGRAPH_HPP

#include <array>
#include <unordered_map>
#include <vector>

// Special hash functor for the EdgeColors unordered_map
// Same as hash_combine from boost
namespace std {
struct ArrayHasher {
  std::size_t operator()(const std::array<int, 2>& a) const {
    std::size_t h = 0;

    for (auto e : a) {
      h ^= std::hash<int>{}(e) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};
}  // namespace std

namespace netket {

/**
    Abstract class for Graphs.
    This class prototypes the methods needed
    by a class satisfying the Graph concept.
    These include lattices and non-regular graphs.
*/
class AbstractGraph {
 public:
  /**
  Member function returning the number of sites (nodes) in the graph.
  @return Number of sites (nodes) in the graph.
  */
  virtual int Nsites() const = 0;
  /**
  Member function returning the integer distances between pair of nodes.
  @return dist[i][j] is the integer distance between site i and j.
  */
  virtual std::vector<std::vector<int>> Distances() const = 0;

  /**
  Member function returning the adjacency list of the graph.
  @return adl[i][k] is the k-th neighbour of site i.
  */
  virtual std::vector<std::vector<int>> AdjacencyList() const = 0;

  /**
  Member function returning the symmetry table of the graph.
  @return st[i][k] contains the i-th equivalent permutation of the sites.
  */
  virtual std::vector<std::vector<int>> SymmetryTable() const = 0;

  /**
  Custom type for unordered_map<array<int,2>, int> w/ a custom hash function
  */
  using ColorMap =
      std::unordered_map<std::array<int, 2>, int, std::ArrayHasher>;

  /**
  Member function returning edge colors of the graph.
  @return ec[i][j] is the color of the edge between nodes i and j.
  */
  virtual ColorMap EdgeColors() const = 0;

  /**
  Member function returning true if the graph is bipartite.
  @return true if lattice is bipartite.
  */
  virtual bool IsBipartite() const = 0;

  virtual ~AbstractGraph(){};
};

}  // namespace netket
#endif
