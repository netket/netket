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

#ifndef NETKET_PYCUSTOMGRAPH_HPP
#define NETKET_PYCUSTOMGRAPH_HPP

#include "custom_graph.hpp"

namespace py = pybind11;
namespace netket {
namespace {
/// Given a Python iterable object constructs the edge list and (optionally)
/// the colour map for the soon to be graph. `callback` is then called in one of
/// the following ways:
/// * `callback(edges, colour_map)` if the iterable contained elements of type
/// `(int, int, int)`.
/// * `callback(edges)` if the iterable contained elements of type `(int, int)`.
template <class Function>
auto WithEdges(py::iterator first, Function&& callback)
    -> decltype(std::forward<Function>(callback)(
        std::declval<std::vector<AbstractGraph::Edge>>())) {
  using std::begin;
  using std::end;

  bool has_colours = false;
  if (first != py::iterator::sentinel()) {
    // We have at least one element, let's determine whether it's an instance
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

  if (has_colours) {
    auto colors = detail::Iterable2ColorMap(first);
    std::vector<AbstractGraph::Edge> edges;
    edges.reserve(colors.size());
    std::transform(
        begin(colors), end(colors), std::back_inserter(edges),
        [](std::pair<AbstractGraph::Edge, int> const& x) { return x.first; });
    return std::forward<Function>(callback)(std::move(edges),
                                            std::move(colors));
  } else {
    return std::forward<Function>(callback)(detail::Iterable2Edges(first));
  }
}

// Work around the lack of C++11 support for defaulted arguments in lambdas.
struct CustomGraphInit {
  using Edge = AbstractGraph::Edge;
  using ColorMap = AbstractGraph::ColorMap;

  std::vector<std::vector<int>> automorphisms;
  bool is_bipartite;

  auto operator()(std::vector<Edge> edges, ColorMap colors = ColorMap{})
      -> std::unique_ptr<CustomGraph> {
    return make_unique<CustomGraph>(std::move(edges), std::move(colors),
                                    std::move(automorphisms), is_bipartite);
  }
};
}  // namespace
}  // namespace netket

namespace netket {

void AddCustomGraph(py::module& subm) {
  py::class_<CustomGraph, AbstractGraph>(subm, "CustomGraph", R"EOF(
      A custom graph, specified by a list of edges and optionally colors.)EOF")
      .def(py::init([](py::iterable xs,
                       std::vector<std::vector<int>> automorphisms,
                       bool const is_bipartite) {
             auto iterator = xs.attr("__iter__")();
             return WithEdges(
                 iterator,
                 CustomGraphInit{std::move(automorphisms), is_bipartite});
           }),
           py::arg("edges"),
           py::arg("automorphisms") = std::vector<std::vector<int>>(),
           py::arg("is_bipartite") = false, R"EOF(
           Constructs a new graph given a list of edges.

           Args:
               edges: If `edges` has elements of type `Tuple[int, int]` it is treated
                   as a list of edges. Then each element `(i, j)` means a connection
                   between sites `i` and `j`. It is assumed that `0 <= i <= j`. Also,
                   `edges` should contain no duplicates. If `edges` has elements of
                   type `Tuple[int, int, int]` each element `(i, j, c)` represents an
                   edge between sites `i` and `j` colored into `c`. It is again assumed
                   that `0 <= i <= j` and that there are no duplicate elements in `edges`.
               automorphisms: The automorphisms of the graph, i.e. a List[List[int]]
                   where the inner List[int] is a unique permutation of the
                   graph sites.
               is_bipartite: Wheter the custom graph is bipartite.
                   Notice that this is not deduced from the edge
                   list and it is left to the user to specify
                   whether the graph is bipartite or not.

           Examples:
               A 10-site one-dimensional lattice with periodic boundary conditions can be
               constructed specifying the edges as follows:

               ```python
               >>> from netket.graph import CustomGraph
               >>> g=CustomGraph([[i, (i + 1) % 10] for i in range(10)])
               >>> print(g.n_sites)
               10

               ```
           )EOF");
}
}  // namespace netket
#endif
