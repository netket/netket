// Copyright 2018 The Simons Foundation, Inc. - All
// Rights Reserved.
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

#ifndef NETKET_ABSTRACT_DENSITY_MATRIX_HPP
#define NETKET_ABSTRACT_DENSITY_MATRIX_HPP

#include "Graph/custom_graph.hpp"
#include "Hilbert/custom_hilbert.hpp"
#include "Machine/abstract_machine.hpp"
#include "Utils/memory_utils.hpp"

namespace netket {

/* Abstract base class for Density Matrices.
 * Contains the physical hilbert space and the doubled hilbert
 * space where operators are defined.
 */
class AbstractDensityMatrix : public AbstractMachine {
  // The physical hilbert space over which this operator acts
  const AbstractHilbert &hilbert_physical_;

  const std::unique_ptr<AbstractGraph> graph_doubled_;

  // The doubled hilbert space where this operator is defined
  const std::unique_ptr<AbstractHilbert> hilbert_doubled_;

  using Edge = AbstractGraph::Edge;

  /**
   * Returns the disjoint union G of a graph g with itself. The resulting graph
   * has twice the number of vertices and edges of g.
   * The automorpisms of G are the automorphisms of g applied identically
   * to both it's subgraphs.
   */
  static std::unique_ptr<CustomGraph> DoubledGraph(const AbstractGraph &graph) {
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
    for (auto automorphism : graph.SymmetryTable()) {
      std::vector<int> d_automorphism = automorphism;
      for (auto s : automorphism) {
        d_automorphism.push_back(s + n_sites);
      }
      d_automorphisms.push_back(d_automorphism);
    }

    return make_unique<CustomGraph>(
        CustomGraph(d_edges, d_eclist, d_automorphisms));
  }

 public:
  explicit AbstractDensityMatrix(const AbstractHilbert &hilbert)
      : hilbert_physical_(hilbert),
        graph_doubled_(DoubledGraph(hilbert.GetGraph())),
        hilbert_doubled_(make_unique<CustomHilbert>(*graph_doubled_,
                                                    hilbert.LocalStates())){};

  const AbstractHilbert &GetHilbert() const noexcept override {
    return *hilbert_doubled_;
  }

  /**
   * Member function returning the Physical hilbert space over which
   * this density matrix acts
   * @return The physical hilbert space
   */
  const AbstractHilbert &GetHilbertPhysical() const noexcept {
    return hilbert_physical_;
  }
};
}  // namespace netket

#endif  // NETKET_ABSTRACT_DENSITY_MATRIX_HPP
