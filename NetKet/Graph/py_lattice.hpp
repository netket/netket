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

#ifndef NETKET_PYLATTICE_HPP
#define NETKET_PYLATTICE_HPP

#include "lattice.hpp"

namespace py = pybind11;

namespace netket {

void AddLattice(py::module& subm) {
  py::class_<Lattice, AbstractGraph>(subm, "Lattice", R"EOF(
                             A generic lattice built translating a unit cell and adding edges between nearest neighbours sites.
                             The unit cell is defined by the ``basis_vectors`` and it can contain an arbitrary number of atoms.
                             Each atom is located at an arbitrary position and is labelled by an integer number,
                             meant to distinguish between the different atoms within the unit cell.
                             Periodic boundary conditions can also be imposed along the desired directions.\n
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
                                 >>> from netket.graph import Lattice
                                 >>> g=Lattice(basis_vectors=[[1,0],[0,1]],extent=[3,4])
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
               >>> from netket.graph import Lattice
               >>> g=Lattice(basis_vectors=[[1.,0.],[0.,1.]], extent=[2,2], pbc=[0,0])
               >>> print(g.site_to_vector(0))
               [0,0]
               >>> print(g.site_to_vector(1))
               [0,1]
               >>> print(g.site_to_vector(2))
               [1,0]
               >>>print(g.site_to_vector(3))
               [1,1]

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
}  // namespace netket
#endif
