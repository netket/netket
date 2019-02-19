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

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "abstract_hilbert.hpp"
#include "py_bosons.hpp"
#include "py_custom_hilbert.hpp"
#include "py_hilbert_index.hpp"
#include "py_qubits.hpp"
#include "py_spins.hpp"

namespace py = pybind11;

namespace netket {

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
      .def("random_vals", &AbstractHilbert::RandomVals, py::arg("state"),
           py::arg("rgen"), R"EOF(
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
           >>> import numpy as np
           >>> hi = nk.hilbert.Boson(n_max=3, graph=nk.graph.Hypercube(length=5, n_dim=1))
           >>> rstate = np.zeros(hi.size)
           >>> rg = nk.utils.RandomEngine(seed=1234)
           >>> hi.random_vals(rstate, rg)
           >>> local_states = hi.local_states
           >>> print(rstate[0] in local_states)
           True

           ```
       )EOF")
      .def("update_conf", &AbstractHilbert::UpdateConf, py::arg("v"),
           py::arg("to_change"), py::arg("new_conf"), R"EOF(
      Member function updating a visible configuration using the information on
      where the local changes have been done.

      Args:
          v: The vector of visible units to be modified.
          to_change: A list of which qunatum numbers will be modified.
          new_conf: Contains the value that those quantum numbers should take.

      )EOF");

  AddSpins(subm);
  AddBosons(subm);
  AddQubits(subm);
  AddCustomHilbert(subm);
  AddHilbertIndex(subm);
}

}  // namespace netket

#endif
