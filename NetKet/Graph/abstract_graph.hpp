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
#include "Utils/array_hasher.hpp"

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
  using Edge = std::array<int, 2>;
  using ColorMap = std::unordered_map<Edge, int, netket::ArrayHasher>;

  /**
  Member function returning edge colors of the graph.
  @return ec[i][j] is the color of the edge between nodes i and j.
  */
  virtual const ColorMap &EdgeColors() const = 0;

  // Edge Colors from users specified map
  void EdgeColorsFromList(const std::vector<std::vector<int>> &colorlist,
                          ColorMap &eclist) {
    for (auto edge : colorlist) {
      eclist[{{edge[0], edge[1]}}] = edge[2];
    }
  }

  // If no Edge Colors are specified, initialize eclist_ with same color (0).
  void EdgeColorsFromAdj(const std::vector<std::vector<int>> &adjlist,
                         ColorMap &eclist) {
    for (int i = 0; i < static_cast<int>(adjlist.size()); i++) {
      for (std::size_t j = 0; j < adjlist[i].size(); j++) {
        eclist[{{i, adjlist[i][j]}}] = 0;
      }
    }
  }

  /**
  Member function returning true if the graph is bipartite.
  @return true if lattice is bipartite.
  */
  virtual bool IsBipartite() const = 0;

  virtual ~AbstractGraph(){};
};

}  // namespace netket
#endif
