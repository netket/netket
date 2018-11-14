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

#ifndef NETKET_PYGRAPH_HPP
#define NETKET_PYGRAPH_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "netket.hpp"

namespace py = pybind11;

namespace netket {

#define ADDGRAPHMETHODS(name)                              \
                                                           \
  .def("Nsites", &AbstractGraph::Nsites)                   \
      .def("AdjacencyList", &AbstractGraph::AdjacencyList) \
      .def("SymmetryTable", &AbstractGraph::SymmetryTable) \
      .def("EdgeColors", &AbstractGraph::EdgeColors)       \
      .def("IsBipartite", &AbstractGraph::IsBipartite)     \
      .def("IsConnected", &AbstractGraph::IsConnected)     \
      .def("Distances", &AbstractGraph::Distances);

/// Given a Python iterable object constructs the edge list and (optionally)
/// the colour map for the soon to be graph. `callback` is then called in one of
/// the following ways:
/// * `callback(edges, colour_map)` if the iterable contained elements of type
/// `(int, int, int)`.
/// * `callback(edges)` if the iterable contained elements of type `(int, int)`.
///
// TODO(twesterhout): We should probably split this into smaller functions...
template <class Function>
auto WithEdges(py::iterable xs, Function&& callback)
    -> decltype(std::forward<Function>(callback)(
        std::declval<std::vector<AbstractGraph::Edge>>())) {
  using std::begin;
  using std::end;
  // Length of some iterables is known in advance, without going through all
  // the elements. We can use this hint to preallocate the edge vector.
  auto const length_hint = [](py::iterable ys) -> std::size_t {
    auto iterator = ys.attr("__iter__")();
    if (py::hasattr(iterator, "__length_hint__")) {
      auto const n = iterator.attr("__length_hint__")().cast<long>();
      assert(n >= 0 && "Bug in Python/pybind11??");
      return static_cast<std::size_t>(n);
    }
    return 0;
  };
  // Correctly orders site indices and constructs an edge.
  // TODO(twesterhout): Should we throw when `x == y`? I.e. edge from a node to
  // itself is a questionable concept.
  auto const make_edge = [](int const x,
                            int const y) noexcept->AbstractGraph::Edge {
    using Edge = AbstractGraph::Edge;
    return (x < y) ? Edge{x, y} : Edge{y, x};
  };

  auto first = begin(xs);
  auto const last = end(xs);
  bool has_colours = false;
  if (first != last) {
    // We have at least one element, let's determine whether it's an intance
    // of `(int, int)` or `(int, int, int)`.
    try {
      // If the following line succeeds, we have a sequence of `(int, int)`.
      static_cast<void>(first->template cast<std::tuple<int, int>>());
      has_colours = false;
    } catch (py::cast_error& /*unused*/) {
      try {
        // If the following line succeeds, we have a sequence of `(int, int,
        // int)`.
        static_cast<void>(first->template cast<std::tuple<int, int, int>>());
        has_colours = true;
      } catch (py::cast_error& /*unused*/) {
        throw py::cast_error("Unable to cast Python instance of type " +
                             std::string{py::str(first->get_type())} +
                             " to either one of the following C++ types: '" +
                             py::type_id<std::tuple<int, int>>() + "', '" +
                             py::type_id<std::tuple<int, int, int>>() + "'");
      }
    }
  }

  auto const hint = length_hint(xs);
  std::vector<AbstractGraph::Edge> edges;
  if (hint != 0) {
    edges.reserve(hint);
  }

  if (has_colours) {
    AbstractGraph::ColorMap colours;
    if (hint != 0) {
      colours.reserve(hint);
    }
    for (; first != last; ++first) {
      auto const x = first->template cast<std::tuple<int, int, int>>();
      auto const edge = make_edge(std::get<0>(x), std::get<1>(x));
      edges.push_back(edge);
      if (!colours.emplace(edge, std::get<2>(x)).second) {
        // Failed to insert an edge because it already exists
        throw InvalidInputError{"Edge list contains duplicates."};
      }
    }
    return std::forward<Function>(callback)(std::move(edges),
                                            std::move(colours));
  }
  // No colours
  for (; first != last; ++first) {
    auto const x = first->template cast<std::tuple<int, int>>();
    edges.push_back(make_edge(std::get<0>(x), std::get<1>(x)));
  }
  // NOTE(twesterhout): Yes, I know that this screws up the algorithmic
  // complexity, but it's fast enough to be unnoticeable for all practical
  // purposes.
  std::sort(begin(edges), end(edges));
  if (std::unique(begin(edges), end(edges)) != end(edges)) {
    throw InvalidInputError{"Edge list contains duplicates."};
  }
  return std::forward<Function>(callback)(std::move(edges));
}

void AddGraphModule(py::module& m) {
  auto subm = m.def_submodule("graph");

  py::class_<AbstractGraph>(subm, "Graph") ADDGRAPHMETHODS(AbstractGraph);

  py::class_<Hypercube, AbstractGraph>(subm, "Hypercube")
      .def(py::init<int, int, bool, std::vector<std::vector<int>>>(),
           py::arg("L"), py::arg("ndim"), py::arg("pbc") = true,
           py::arg("edgecolors") = std::vector<std::vector<int>>())
      .def("Nsites", &Hypercube::Nsites)
      .def("AdjacencyList", &Hypercube::AdjacencyList)
      .def("SymmetryTable", &Hypercube::SymmetryTable)
      .def("EdgeColors", &Hypercube::EdgeColors)
      .def("IsBipartite", &Hypercube::IsBipartite)
      .def("IsConnected", &Hypercube::IsConnected)
      .def("Distances", &Hypercube::Distances)
      .def("AllDistances", &Hypercube::AllDistances) ADDGRAPHMETHODS(Hypercube);

  py::class_<CustomGraph, AbstractGraph>(subm, "CustomGraph")
#if 0  // TODO(twesterhout): Remove completely
      .def(
          py::init<int, std::vector<std::vector<int>>,
                   std::vector<std::vector<int>>, std::vector<std::vector<int>>,
                   std::vector<std::vector<int>>, bool>(),
          py::arg("size") = 0,
          py::arg("adjacency_list") = std::vector<std::vector<int>>(),
          py::arg("edges") = std::vector<std::vector<int>>(),
          py::arg("automorphisms") = std::vector<std::vector<int>>(),
          py::arg("edgecolors") = std::vector<std::vector<int>>(),
          py::arg("is_bipartite") = false)
#endif
      .def(py::init([](py::iterable xs,
                       std::vector<std::vector<int>> automorphisms =
                           std::vector<std::vector<int>>(),
                       bool const is_bipartite = false) {
             using Edge = AbstractGraph::Edge;
             using ColorMap = AbstractGraph::ColorMap;
             return WithEdges(xs, [&automorphisms, is_bipartite](
                                      std::vector<Edge> edges,
                                      ColorMap colors = ColorMap{}) {
               return make_unique<CustomGraph>(
                   std::move(edges), std::move(colors),
                   std::move(automorphisms), is_bipartite);
             });
           }),
           py::arg("edges"),
           py::arg("automorphisms") = std::vector<std::vector<int>>(),
           py::arg("is_bipartite") = false,
           R"mydelimiter(
               Constructs a new graph given a list of edges.

                   * If `edges` has elements of type `Tuple[int, int]` it is treated
                     as a list of edges. Then each element `(i, j)` means a connection
                     between sites `i` and `j`. It is assumed that `0 <= i <= j`. Also,
                     `edges` should contain no duplicates.

                   * If `edges` has elements of type `Tuple[int, int, int]` each
                     element `(i, j, c)` represents an edge between sites `i` and `j`
                     colored into `c`. It is again assumed that `0 <= i <= j` and that
                     there are no duplicate elements in `edges`.
          )mydelimiter")
      .def("Nsites", &CustomGraph::Nsites)
      .def("AdjacencyList", &CustomGraph::AdjacencyList)
      .def("SymmetryTable", &CustomGraph::SymmetryTable)
      .def("EdgeColors", &CustomGraph::EdgeColors)
      .def("IsBipartite", &CustomGraph::IsBipartite)
      .def("IsConnected", &CustomGraph::IsConnected)
      .def("Distances", &CustomGraph::Distances)
      .def("AllDistances", &CustomGraph::AllDistances)
          ADDGRAPHMETHODS(CustomGraph);
}

}  // namespace netket

#endif
