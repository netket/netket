//
// Created by Filippo Vicentini on 12/11/2019.
//

#include "doubled_graph.hpp"

#include "Utils/memory_utils.hpp"

namespace netket {
using Edge = AbstractGraph::Edge;

/**
 * Returns the disjoint union G of a graph g with itself. The resulting graph
 * has twice the number of vertices and edges of g.
 * The automorpisms of G are the automorphisms of g applied identically
 * to both it's subgraphs.
 */
std::unique_ptr<CustomGraph> DoubledGraph(const AbstractGraph &graph) {
  auto n_sites = graph.Nsites();
  std::vector<Edge> d_edges(graph.Edges().size());
  auto eclist = graph.EdgeColors();

  // same graph
  auto d_eclist = graph.EdgeColors();
  for (auto edge : graph.Edges()) {
    d_edges.push_back(edge);
  }

  // second graph
  for (auto edge : graph.Edges()) {
    Edge new_edge = edge;
    new_edge[0] += n_sites;
    new_edge[1] += n_sites;

    d_edges.push_back(new_edge);
    d_eclist.emplace(new_edge, eclist[edge]);
  }

  std::vector<std::vector<int>> d_automorphisms;
  for (const auto &automorphism : graph.SymmetryTable()) {
    std::vector<int> d_automorphism = automorphism;
    for (auto s : automorphism) {
      d_automorphism.push_back(s + n_sites);
    }
    d_automorphisms.push_back(d_automorphism);
  }

  return make_unique<CustomGraph>(
      CustomGraph(d_edges, d_eclist, d_automorphisms));
}

}  // namespace netket