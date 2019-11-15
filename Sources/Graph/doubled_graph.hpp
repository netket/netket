//
// Created by Filippo Vicentini on 12/11/2019.
//

#ifndef NETKET_DOUBLED_GRAPH_HPP
#define NETKET_DOUBLED_GRAPH_HPP

#include <memory>

#include "Graph/custom_graph.hpp"
#include "Graph/edgeless.hpp"

namespace netket {
/**
 * Returns the disjoint union G of a graph g with itself. The resulting graph
 * has twice the number of vertices and edges of g.
 * The automorpisms of G are the automorphisms of g applied identically
 * to both it's subgraphs.
 */
std::unique_ptr<AbstractGraph> DoubledGraph(const AbstractGraph &graph);
}  // namespace netket
#endif  // NETKET_DOUBLED_GRAPH_HPP
