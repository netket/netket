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

#include "py_graph.hpp"

#include "Utils/memory_utils.hpp"
#include "abstract_graph.hpp"
#include "custom_graph.hpp"
#include "hypercube.hpp"
#include "lattice.hpp"

namespace py = pybind11;

namespace netket {

namespace detail {
namespace {
#if 0
/// Given a Python iterable, returns its length if it is known or 0 otherwise.
/// This can safely be used to preallocate storage on the C++ side as calls to
/// `std::vector<T>::reserve(0)` are basically noops.
inline std::size_t LengthHint(py::iterable xs) {
  auto iterator = xs.attr("__iter__")();
  if (py::hasattr(iterator, "__length_hint__")) {
    auto const n = iterator.attr("__length_hint__")().cast<long>();
    assert(n >= 0 && "Bug in Python/pybind11??");
    return static_cast<std::size_t>(n);
  }
  return 0;
}
#endif

inline std::size_t LengthHint(py::iterator x) {
  if (py::hasattr(x, "__length_hint__")) {
    auto const n = x.attr("__length_hint__")().cast<long>();
    assert(n >= 0 && "Bug in Python/pybind11??");
    return static_cast<std::size_t>(n);
  }
  return 0;
}

/// Correctly orders site indices and constructs an edge.
// TODO(twesterhout): Should we throw when `x == y`? I.e. edge from a node to
// itself is a questionable concept.
inline AbstractGraph::Edge MakeEdge(int const x, int const y) noexcept {
  using Edge = AbstractGraph::Edge;
  return (x < y) ? Edge{{x, y}} : Edge{{y, x}};
}

/// Converts a Python iterable to a list of edges. An exception is thrown if the
/// input iterable contains duplicate edges.
///
/// \postcondition For each edge (i, j) we have i <= j.
/// \postcondition The returned list contains no duplicates.
inline std::vector<AbstractGraph::Edge> Iterable2Edges(py::iterator x) {
  using std::begin;
  using std::end;
  std::vector<AbstractGraph::Edge> edges;
  edges.reserve(LengthHint(x));

  while (x != py::iterator::sentinel()) {
    int i, j;
    std::tie(i, j) = x->template cast<std::tuple<int, int>>();
    edges.push_back(MakeEdge(i, j));
    ++x;
  }

  // NOTE(twesterhout): Yes, I know that this screws up the algorithmic
  // complexity, but it's fast enough to be unnoticeable for all practical
  // purposes.
  std::sort(begin(edges), end(edges));
  if (std::unique(begin(edges), end(edges)) != end(edges)) {
    throw InvalidInputError{"Edge list contains duplicates."};
  }
  return edges;
}

/// Converts a Python iterable to a `ColorMap`. An exception is thrown if the
/// input iterable contains duplicate edges.
///
/// \postcondition For each edge (i, j) we have i <= j.
inline AbstractGraph::ColorMap Iterable2ColorMap(py::iterator x) {
  AbstractGraph::ColorMap colors;
  colors.reserve(LengthHint(x));

  while (x != py::iterator::sentinel()) {
    int i, j, color;
    std::tie(i, j, color) = x->template cast<std::tuple<int, int, int>>();
    if (!colors.emplace(MakeEdge(i, j), color).second) {
      // Failed to insert an edge because it already exists
      throw InvalidInputError{"Edge list contains duplicates."};
    }
    ++x;
  }
  return colors;
}
}  // namespace
}  // namespace detail

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

  auto operator()(std::vector<Edge> edges, ColorMap colors = ColorMap{})
      -> std::unique_ptr<CustomGraph> {
    return make_unique<CustomGraph>(std::move(edges), std::move(colors),
                                    std::move(automorphisms));
  }
};

void AddAbstractGraph(py::module subm) {
  py::class_<AbstractGraph>(subm, "Graph")
      .def_property_readonly("n_sites", &AbstractGraph::Nsites,
                             R"EOF(
      int: The number of vertices in the graph.)EOF")
      .def_property_readonly(
          "edges",
          [](AbstractGraph const& x) {
            using vector_type =
                std::remove_reference<decltype(x.Edges())>::type;
            return vector_type{x.Edges()};
          },
          R"EOF(
      list: The graph edges.)EOF")
      .def_property_readonly("adjacency_list", &AbstractGraph::AdjacencyList,
                             R"EOF(
      list: The adjacency list of the graph where each node is
          represented by an integer in `[0, n_sites)`.)EOF")
      .def_property_readonly("is_bipartite", &AbstractGraph::IsBipartite,
                             R"EOF(
      bool: Whether the graph is bipartite.)EOF")
      .def_property_readonly("is_connected", &AbstractGraph::IsConnected,
                             R"EOF(
      bool: Whether the graph is connected.)EOF")
      .def_property_readonly("distances", &AbstractGraph::AllDistances,
                             R"EOF(
      list[list]: The distances between the nodes. The fact that some node
          may not be reachable from another is represented by -1.)EOF")
      .def_property_readonly("automorphisms", &AbstractGraph::SymmetryTable,
                             R"EOF(
      list[list]: The automorphisms of the graph,
          including translation symmetries only.)EOF");
}

void AddCustomGraph(py::module subm) {
  py::class_<CustomGraph, AbstractGraph>(subm, "CustomGraph", R"EOF(
      A custom graph, specified by a list of edges and optionally colors.)EOF")
      .def(py::init([](py::iterable xs,
                       std::vector<std::vector<int>> automorphisms) {
             auto iterator = xs.attr("__iter__")();
             return WithEdges(iterator,
                              CustomGraphInit{std::move(automorphisms)});
           }),
           py::arg("edges"),
           py::arg("automorphisms") = std::vector<std::vector<int>>(), R"EOF(
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


           Examples:
               A 10-site one-dimensional lattice with periodic boundary conditions can be
               constructed specifying the edges as follows:

               ```python
               >>> import netket
               >>> g=netket.graph.CustomGraph([[i, (i + 1) % 10] for i in range(10)])
               >>> print(g.n_sites)
               10

               ```
           )EOF");
}

void AddHypercube(py::module subm) {
  py::class_<Hypercube, AbstractGraph>(subm, "Hypercube",
                                       R"EOF(
         A hypercube lattice of side L in d dimensions.
         Periodic boundary conditions can also be imposed.)EOF")
      .def(py::init<int, int, bool>(), py::arg("length"), py::arg("n_dim") = 1,
           py::arg("pbc") = true, R"EOF(
         Constructs a new ``Hypercube`` given its side length and dimension.

         Args:
             length: Side length of the hypercube.
                 It must always be >=1,
                 but if ``pbc==True`` then the minimal
                 valid length is 3.
             n_dim: Dimension of the hypercube. It must be at least 1.
             pbc: If ``True`` then the constructed hypercube
                 will have periodic boundary conditions, otherwise
                 open boundary conditions are imposed.

         Examples:
             A 10x10 square lattice with periodic boundary conditions can be
             constructed as follows:

             ```python
             >>> import netket
             >>> g=netket.graph.Hypercube(length=10,n_dim=2,pbc=True)
             >>> print(g.n_sites)
             100

             ```
         )EOF")
      .def(py::init([](int length, py::iterable xs) {
             auto iterator = xs.attr("__iter__")();
             return Hypercube{length, detail::Iterable2ColorMap(iterator)};
           }),
           py::arg("length"), py::arg("colors"), R"EOF(
         Constructs a new `Hypercube` given its side length and edge coloring.

         Args:
             length: Side length of the hypercube.
                 It must always be >=3 if the
                 hypercube has periodic boundary conditions
                 and >=1 otherwise.
             colors: Edge colors, must be an iterable of
                 `Tuple[int, int, int]` where each
                 element `(i, j, c) represents an
                 edge `i <-> j` of color `c`.
                 Colors must be assigned to **all** edges.)EOF");
}

void AddLattice(py::module subm) {
  py::class_<Lattice, AbstractGraph>(subm, "Lattice", R"EOF(
                             A generic lattice built translating a unit cell and adding edges between nearest neighbours sites.
                             The unit cell is defined by the ``basis_vectors`` and it can contain an arbitrary number of atoms.
                             Each atom is located at an arbitrary position and is labelled by an integer number,
                             meant to distinguish between the different atoms within the unit cell.
                             Periodic boundary conditions can also be imposed along the desired directions.
                             There are three different ways to refer to the lattice sites. A site can be labelled
                             by a simple integer number (the site index), by its coordinates (actual position in space),
                             or by a set of integers (the site vector), which indicates how many
                             translations of each basis vectors have been performed while building the
                             graph. The i-th component refers to translations along the i-th ``basis_vector`` direction.)EOF")
      .def(py::init<std::vector<std::vector<double>>, std::vector<int>,
                    std::vector<bool>, std::vector<std::vector<double>>>(),
           py::arg("basis_vectors"), py::arg("extent"),
           py::arg("pbc") = std::vector<double>(0),
           py::arg("atoms_coord") = std::vector<std::vector<double>>(0),
           R"EOF(
                             Constructs a new ``Lattice`` given its side length and the features of the unit cell.

                             Args:
                                 basis_vectors: The basis vectors of the unit cell.
                                 extent: The number of copies of the unit cell.
                                 pbc: If ``True`` then the constructed lattice
                                     will have periodic boundary conditions, otherwise
                                     open boundary conditions are imposed (default=``True``).
                                 atoms_coord: The coordinates of different atoms in the unit cell (default=one atom at the origin).

                             Examples:
                                 Constructs a rectangular 3X4 lattice with periodic boundary conditions.

                                 ```python
                                 >>> import netket
                                 >>> g=netket.graph.Lattice(basis_vectors=[[1,0],[0,1]],extent=[3,4])
                                 >>> print(g.n_sites)
                                 12

                                 ```
                             )EOF")
      .def_property_readonly("coordinates", &Lattice::Coordinates,
                             R"EOF(
      list[list]: The coordinates of the atoms in the lattice.)EOF")
      .def_property_readonly("n_dim", &Lattice::Ndim, R"EOF(
      int: The dimension of the lattice.)EOF")
      .def_property_readonly("basis_vectors", &Lattice::BasisVectors, R"EOF(
      list[list]: The basis vectors of the lattice.)EOF")
      .def("atom_label", &Lattice::AtomLabel, py::arg("site"), R"EOF(
          Member function returning the atom label indicating which of the unit cell atoms is located at a given a site index.

          Args:
              site: The site index.

          )EOF")
      .def("site_to_vector", &Lattice::Site2Vector, py::arg("site"), R"EOF(
          Member function returning the site vector corresponding to a given site index.

          Args:
              site: The site index.

          Examples:
              Constructs a square 2X2 lattice without periodic boundary conditions and prints the site vectors corresponding to given site indices.

              ```python
               >>> import netket
               >>> g=netket.graph.Lattice(basis_vectors=[[1.,0.],[0.,1.]], extent=[2,2], pbc=[0,0])
               >>> print(list(map(int,g.site_to_vector(0))))
               [0, 0]
               >>> print(list(map(int,g.site_to_vector(1))))
               [0, 1]
               >>> print(list(map(int,g.site_to_vector(2))))
               [1, 0]
               >>> print(list(map(int,g.site_to_vector(3))))
               [1, 1]

                ```
            )EOF")
      .def("vector_to_coord", &Lattice::Vector2Coord, py::arg("site_vector"),
           py::arg("atom_label"), R"EOF(
        Member function returning the coordinates of a given atom characterized by a
        given site vector.

          Args:
              site_vector: The site vector.
              atom_label: Which of the atoms in the unit cell.

            )EOF")
      .def("site_to_coord", &Lattice::Site2Coord, py::arg("site"), R"EOF(
        Member function returning the coordinates of a given site index.

          Args:
              site: The site index.
            )EOF")
      .def("vector_to_site", &Lattice::Vector2Site, py::arg("site_vector"),
           R"EOF(
        Member function returning the site index corresponding to a given site vector.

            Args:
                site_vector: The site vector.
            )EOF");
}
}  // namespace

void AddGraphModule(py::module m) {
  auto subm = m.def_submodule("graph");

  AddAbstractGraph(subm);
  AddHypercube(subm);
  AddCustomGraph(subm);
  AddLattice(subm);
}

}  // namespace netket
