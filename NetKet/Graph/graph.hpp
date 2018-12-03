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

#ifndef NETKET_GRAPH_HPP
#define NETKET_GRAPH_HPP

#include <array>
#include <unordered_map>
#include <vector>
#include "Utils/json_utils.hpp"
#include "Utils/memory_utils.hpp"
#include "abstract_graph.hpp"
#include "custom_graph.hpp"
#include "hypercube.hpp"
#include "mpark/variant.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace netket {
class Graph : public AbstractGraph {
 public:
  using VariantType = mpark::variant<Hypercube, CustomGraph>;

 private:
  VariantType obj_;

 public:
  Graph(VariantType obj) : obj_(obj) {}

  int Nsites() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.Nsites(); }, obj_);
  }

  int Size() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.Size(); }, obj_);
  }

  std::vector<Edge> const Edges() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.Edges(); }, obj_);
  }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return mpark::visit([](auto &&obj) { return obj.AdjacencyList(); }, obj_);
  }

  std::vector<std::vector<int>> SymmetryTable() const override {
    return mpark::visit([](auto &&obj) { return obj.SymmetryTable(); }, obj_);
  }

  const ColorMap EdgeColors() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.EdgeColors(); }, obj_);
  }

  bool IsBipartite() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.IsBipartite(); }, obj_);
  }

  bool IsConnected() const noexcept override {
    return mpark::visit([](auto &&obj) { return obj.IsConnected(); }, obj_);
  }

  std::vector<int> Distances(int root) const override {
    return mpark::visit([root](auto &&obj) { return obj.Distances(root); },
                        obj_);
  }

  std::vector<std::vector<int>> AllDistances() const override {
    return mpark::visit([](auto &&obj) { return obj.AllDistances(); }, obj_);
  }
};
}  // namespace netket
#endif
