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

#ifndef NETKET_PYHILBERT_HPP
#define NETKET_PYHILBERT_HPP

#include "hilbert.hpp"
#include <complex>
#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

namespace netket {

constexpr int HilbertIndex::MaxStates;

void AddHilbertModule(py::module &m) {
  auto subm = m.def_submodule("hilbert");

  py::class_<AbstractHilbert>(subm, "Hilbert")
      .def_property_readonly(
          "is_discrete", &AbstractHilbert::IsDiscrete,
          R"EOF(bool: Whether the hilbert space is discrete.)EOF")
      .def_property_readonly("local_size", &AbstractHilbert::LocalSize,
                             R"EOF(int: Size of the local hilbert space.)EOF")
      .def_property_readonly(
          "size", &AbstractHilbert::Size,
          R"EOF(int: The number of visible units needed to describe the system.)EOF")
      .def_property_readonly(
          "local_states", &AbstractHilbert::LocalStates,
          R"EOF(list[float]: List of discreet local quantum numbers.)EOF")
      .def("random_vals", &AbstractHilbert::RandomVals, R"EOF(
       Member function generating uniformely distributed local random states.

       Args:
           state: A reference to a visible configuration, in output this 
               contains the random state.
           rgen: The random number generator.

       Examples:
           Test that a new random state is a possible state for the hilbert
           space.

           ```python
           >>> import netket as nk
           >>> nk.hilbert.Boson(n_max=3, graph=nk.graph.Hypercube(length=5, n_dim=1)
           >>> rstate = np.zeros(hi.size)
           >>> rg = nk.utils.RandomEngine(seed=1234)
           >>> hi.random_vals(rstate, rg)
           >>> local_states = hi.local_states
           >>> print(rstate[0] in local_states)
           True

       ```
       )EOF")
      .def("update_conf", &AbstractHilbert::UpdateConf, R"EOF(
      Member function updating a visible configuration using the information on
      where the local changes have been done.

      Ars:
          v: The vector of visible units to be modified.
          tochange: A list of which qunatum numbers will be modified.
          newconf: Contains the value that those quantum numbers should take.

      )EOF");

  py::class_<Spin, AbstractHilbert>(
      subm, "Spin", R"EOF(Hilbert space composed of spin states.)EOF")
      .def(py::init<const AbstractGraph &, double>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("s"), R"EOF(
           Constructs a new ``Spin`` given a graph and the value of each spin.

           Args:
               graph: Graph representation of sites.
               s: Spin at each site. Must be integer or half-integer.

           Examples:
               Simple spin hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Spin
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Spin(graph=g, s=0.5)
               >>> print(hi.size)
               100

               ```
           )EOF")
      .def(py::init<const AbstractGraph &, double, double>(),
           py::keep_alive<1, 2>(), py::arg("graph"), py::arg("s"),
           py::arg("total_sz"), R"EOF(
           Constructs a new ``Spin`` given a graph and the value of each spin.

           Args:
               graph: Graph representation of sites.
               s: Spin at each site. Must be integer or half-integer.
               total_sz: Constrain total spin of system to a particular value.

           Examples:
               Simple spin hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Spin
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Spin(graph=g, s=0.5, total_sz=0)
               >>> print(hi.size)
               100

               ```
           )EOF");

  py::class_<Qubit, AbstractHilbert>(
      subm, "Qubit", R"EOF(Hilbert space composed of qubits.)EOF")
      .def(py::init<const AbstractGraph &>(), py::keep_alive<1, 2>(),
           py::arg("graph"), R"EOF(
           Constructs a new ``Qubit`` given a graph.

           Args:
               graph: Graph representation of sites.

           Examples:
               Simple qubit hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Qubit
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Qubit(graph=g)
               >>> print(hi.size)
               100

               ```
           )EOF");

  py::class_<Boson, AbstractHilbert>(
      subm, "Boson", R"EOF(Hilbert space composed of bosonic states.)EOF")
      .def(py::init<const AbstractGraph &, int>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("n_max"), R"EOF(
           Constructs a new ``Boson`` given a graph and maximum occupation number.

           Args:
               graph: Graph representation of sites.
               n_max: Maximum occupation for a site.

           Examples:
               Simple boson hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Boson
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Boson(graph=g, n_max=4)
               >>> print(hi.size)
               100

               ```
           )EOF")
      .def(py::init<const AbstractGraph &, int, int>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("n_max"), py::arg("n_bosons"), R"EOF(
           Constructs a new ``Boson`` given a graph,  maximum occupation number,
           and total number of bosons.

           Args:
               graph: Graph representation of sites.
               n_max: Maximum occupation for a site.
               n_bosons: Constraint for the number of bosons.

           Examples:
               Simple boson hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Boson
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Boson(graph=g, n_max=5, n_bosons=11)
               >>> print(hi.size)
               100

               ```
           )EOF");

  py::class_<CustomHilbert, AbstractHilbert>(subm, "CustomHilbert",
                                             R"EOF(A custom hilbert space.)EOF")
      .def(py::init<const AbstractGraph &, std::vector<double>>(),
           py::keep_alive<1, 2>(), py::arg("graph"), py::arg("local_states"),
           R"EOF(
           Constructs a new ``CustomHilbert`` given a graph and a list of 
           eigenvalues of the states.

           Args:
               graph: Graph representation of sites.
               local_states: Eigenvalues of the states.

           Examples:
               Simple custom hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import CustomHilbert
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = CustomHilbert(graph=g, local_states=[-1232, 132, 0])
               >>> print(hi.size)
               100

               ```
           )EOF");

  py::class_<HilbertIndex>(subm, "HilbertIndex")
      .def(py::init<const AbstractHilbert &>(), py::arg("hilbert"))
      .def_property_readonly("n_states", &HilbertIndex::NStates)
      .def("number_to_state", &HilbertIndex::NumberToState)
      .def("state_to_number", &HilbertIndex::StateToNumber)
      .def_readonly_static("max_states", &HilbertIndex::MaxStates);

} // namespace netket

} // namespace netket

#endif
