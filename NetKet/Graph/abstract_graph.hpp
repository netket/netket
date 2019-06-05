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

#ifndef NETKET_ABSTRACT_GRAPH_HPP
#define NETKET_ABSTRACT_GRAPH_HPP

#include <array>
#include <unordered_map>
#include <vector>

#include "Utils/array_hasher.hpp"

namespace netket {

/**
    Abstract class for undirected Graphs.
    This class prototypes the methods needed
    by a class satisfying the Graph concept.
    These include lattices and non-regular graphs.
*/
class AbstractGraph {
 public:
  /**
  Custom type for unordered_map<array<int,2>, int> w/ a custom hash function
  */
  using Edge = std::array<int, 2>;
  using ColorMap = std::unordered_map<Edge, int, netket::ArrayHasher>;

  /**
  Member function returning the number of sites (nodes) in the graph.
  @return Number of sites (nodes) in the graph.
  */
  virtual int Nsites() const noexcept = 0;

  /**
  Member function returning the number of sites (nodes) in the graph.
  @return Number of sites (nodes) in the graph.
  */
  virtual int Size() const noexcept = 0;

  /**
  Member function returning the graph edges.
  @return Graph edges.
  */
  virtual std::vector<Edge> const &Edges() const noexcept = 0;

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
  Member function returning edge colors of the graph.
  @return ec[i][j] is the color of the edge between nodes i and j.
  */
  virtual const ColorMap &EdgeColors() const noexcept = 0;

 protected:
  // Edge Colors from users specified map
  void EdgeColorsFromList(const std::vector<std::vector<int>> &colorlist,
                          ColorMap &eclist);

  // If no Edge Colors are specified, initialize eclist_ with same color (0).
  void EdgeColorsFromAdj(const std::vector<std::vector<int>> &adjlist,
                         ColorMap &eclist);

 public:
  /**
  Member function returning true if the graph is bipartite.
  @return true if lattice is bipartite.
  */
  // virtual bool IsBipartite() const noexcept = 0;
  virtual bool IsBipartite() const noexcept;

  /**
   * Checks whether the graph is connected, i.e., there exists a path between
   * every pair of nodes.
   * @return true, if the graph is connected
   */
  // virtual bool IsConnected() const noexcept = 0;
  virtual bool IsConnected() const noexcept;

  /**
   * Perform a breadth-first search (BFS) through the graph, calling
   * visitor_func exactly once for each node within the component reachable from
   * start.
   * @param start The starting node for the BFS.
   * @param visitor_func Function void visitor_func(int node, int depth) which
   * is called once for each visited node and where depth is the distance of
   * node from start.
   */
  template <typename Func>
  void BreadthFirstSearch(int start, Func visitor_func) const;

  /**
   * Perform a breadth-first search (BFS) through the graph, calling
   * visitor_func exactly once for each visited node. The search will visit
   * all nodes reachable from start in at most max_depth steps.
   * @param start The starting node for the BFS.
   * @param max_depth The maximum distance from start for nodes to be visited.
   * @param visitor_func Function void visitor_func(int node, int depth) which
   * is called once for each visited node and where depth is the distance of
   * node from start.
   */
  template <typename Func>
  void BreadthFirstSearch(int start, int max_depth, Func visitor_func) const;

  /**
   * Perform a breadth-first search (BFS) through the whole graph, calling
   * visitor_func exactly once for each node.
   *
   * If the graph is not connected, the BFS will first explore starting from
   * the node with index 0 and explore the corresponding component.
   * Then, it will iterate over all remaining unexplored nodes, exploring their
   * components in turn until all nodes have been visited.
   *
   * @param visitor_func Function with signature
   *        void visitor_func(int node, int depth, int comp)
   *    which is called once for each visited node. The parameter comp
   *    is the index of the first node within the connected component currently
   *    explored, which allows to distinguish the components. The depth is the
   *    distance from comp to the current node.
   */
  template <typename Func>
  void BreadthFirstSearch(Func visitor_func) const;

  /**
   * Computes the distances of all nodes from a root node (single-source
   * shortest path problem). The distance of nodes not reachable from root are
   * set to -1.
   * @param root The root node from which the distances are calculated.
   * @return A vector `dist` of distances where dist[v] is the distance of v to
   * root.
   */
  virtual std::vector<int> Distances(int root) const;

  /**
   * Computes the distances between each pair of nodes on the graph (the
   * all-pairs shortest path problem).
   * @return A vector of vectors dist where dist[v][w] is the distance between
   *    nodes v and w.
   */
  virtual std::vector<std::vector<int>> AllDistances() const;

  virtual ~AbstractGraph(){};

 protected:
  /**
   * Implementation function for breath-first search.
   * This takes @param seen as a reference, so that the list of already visited
   * vertices can be reused if needed.
   * @param seen needs to satisfy seen.size() == Nsites(). Nodes v where
   * seen[v] == true will be ignored even when they are first discovered
   * by the search. seen[start] is required to be false.
   */
  template <typename Func>
  void BreadthFirstSearch_Impl(int start, int max_depth, Func visitor_func,
                               std::vector<bool> &seen) const;
};

namespace detail {
std::vector<std::vector<int>> AdjacencyListFromEdges(
    const std::vector<AbstractGraph::Edge> &edges, int const number_sites);

int CheckEdges(std::vector<AbstractGraph::Edge> const &edges);
}  // namespace detail

}  // namespace netket

// Implementation of template functions
#include "abstract_graph.ipp"

#endif  // NETKET_ABSTRACT_GRAPH_HPP
