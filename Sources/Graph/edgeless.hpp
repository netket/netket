// Copyright 2018-2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#ifndef NETKET_EDGELESS_HPP
#define NETKET_EDGELESS_HPP

#include "abstract_graph.hpp"

namespace netket {

/**
    Edgeless graph (only vertices without edges)
*/
class Edgeless : public AbstractGraph {
 public:
  using AbstractGraph::ColorMap;
  using AbstractGraph::Edge;

 private:
  int n_sites_;  ///< Total number of nodes in the graph
  std::vector<std::vector<int>> automorphisms_;
  std::vector<Edge> edges_;
  ColorMap cmap_;

 public:
  Edgeless(int n_vertices);

  int Nsites() const noexcept override;
  int Size() const noexcept override;
  std::vector<Edge> const &Edges() const noexcept override;
  std::vector<std::vector<int>> AdjacencyList() const override;
  const ColorMap &EdgeColors() const noexcept override;
  std::vector<std::vector<int>> SymmetryTable() const override;
  bool IsConnected() const noexcept override;
  bool IsBipartite() const noexcept override;
};

}  // namespace netket

#endif  // NETKET_CUSTOM_GRAPH_HPP
