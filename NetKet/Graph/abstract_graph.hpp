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

#include <cassert>
#include <queue>
#include <utility>
#include <vector>

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
  Member function returning true if the graph is bipartite.
  @return true if lattice is bipartite.
  */
  virtual bool IsBipartite() const = 0;

  /**
   * Perform a breadth-first search (BFS) through the graph, calling
   * visitor_func exactly once for each visited node. The search will visit
   * all nodes reachable from start.
   * @param start The starting node for the BFS.
   * @param visitor_func Function void visitor_func(int node, int depth) which is
   *    called once for each visited node and where depth is the distance of node from start.
   */
  template<typename Func>
  void BreadthFirstSearch(int start, Func visitor_func) const {
    assert(start >= 0 && start < Nsites());
    BreadthFirstSearch(start, Nsites(), visitor_func);
  }

  /**
   * Perform a breadth-first search (BFS) through the graph, calling
   * visitor_func exactly once for each visited node. The search will visit
   * all nodes reachable from start in at most max_depth steps.
   * @param start The starting node for the BFS.
   * @param max_depth The maximum distance from start for nodes to be visited.
   * @param visitor_func Function void visitor_func(int node, int depth) which is
   *    called once for each visited node and where depth is the distance of node from start.
   */
  template<typename Func>
  void BreadthFirstSearch(int start, int max_depth, Func visitor_func) const {
    // Pair of node and depth
    using QueueEntry = std::pair<int, int>;

    assert(start >= 0 && start < Nsites());
    assert(max_depth > 0);

    const auto adjacency_list = AdjacencyList();

    // Store the already seen sites
    std::vector<bool> seen(Nsites());
    std::fill(seen.begin(), seen.end(), false);
    seen[start] = true;

    // Queue to store states to visit
    std::queue<QueueEntry> queue;
    queue.push({start, 0});

    while (!queue.empty()) {
      const auto elem = queue.front();
      queue.pop();
      const int node = elem.first;
      const int depth = elem.second;

      if (depth > max_depth) {
        continue;
      }

      visitor_func(node, depth);

      for (const int adj : adjacency_list[node]) {
        if (!seen[adj]) {
          seen[adj] = true;
          queue.push({adj, depth + 1});
        }
      }
    }
  }

  virtual ~AbstractGraph(){};
};

}  // namespace netket
#endif
