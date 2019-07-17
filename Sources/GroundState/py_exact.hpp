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

#ifndef NETKET_PYEXACT_HPP
#define NETKET_PYEXACT_HPP

#include <complex>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "Dynamics/exact_time_propagation.hpp"

namespace py = pybind11;

namespace netket {

void AddExactModule(py::module &m) {
  auto m_exact = m.def_submodule("exact");

  py::class_<ExactTimePropagation>(
      m_exact, "ExactTimePropagation",
      R"EOF(Solving for the ground state of the wavefunction using imaginary time propagation.)EOF")
      .def(py::init<const AbstractOperator &, ExactTimePropagation::Stepper &,
                    double, ExactTimePropagation::StateVector,
                    const std::string &, const std::string &>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("hamiltonian"), py::arg("stepper"),
           py::arg("t0"), py::arg("initial_state"),
           py::arg("matrix_type") = "sparse",
           py::arg("propagation_type") = "real", R"EOF(
           Constructs an ``ExactTimePropagation`` object from a hamiltonian, a stepper,
           a time, and an initial state.

           Args:
               hamiltonian: The hamiltonian of the system.
               stepper: Stepper (i.e. propagator) that transforms the state of
                   the system from one timestep to the next.
               t0: The initial time.
               initial_state: The initial state of the system (when propagation
                   begins.)
               matrix_type: The type of matrix used for the Hamiltonian when
                   creating the matrix wrapper. The default is `sparse`. The
                   other choices are `dense` and `direct`.
               propagation_type: Specifies whether the imaginary or real-time
                   Schroedinger equation is solved. Should be one of "real" or
                   "imaginary".

           Examples:
               Solving 1D Ising model with imaginary time propagation:

               ```python
               >>> import netket as nk
               >>> import numpy as np
               >>> L = 8
               >>> graph = nk.graph.Hypercube(L, n_dim=1, pbc=True)
               >>> hilbert = nk.hilbert.Spin(graph, 0.5)
               >>> n_states = hilbert.n_states
               >>> hamiltonian = nk.operator.Ising(hilbert, h=1.0)
               >>> stepper = nk.dynamics.timestepper(n_states, rel_tol=1e-10, abs_tol=1e-10)
               >>> output = nk.output.JsonOutputWriter('test.log', 'test.wf')
               >>> psi0 = np.random.rand(n_states)
               >>> driver = nk.exact.ExactTimePropagation(hamiltonian, stepper, t0=0,
               ...                                        initial_state=psi0,
               ...                                        propagation_type="imaginary")
               >>> driver.add_observable(hamiltonian, 'Hamiltonian')
               >>> for step in driver.iter(dt=0.05, n_iter=20):
               ...     obs = driver.get_observable_stats()

               ```
           )EOF")
      .def("add_observable", &ExactTimePropagation::AddObservable,
           py::keep_alive<1, 2>(), py::arg("observable"), py::arg("name"),
           py::arg("matrix_type") = "sparse", R"EOF(
           Add an observable quantity, that will be calculated at each
           iteration.

           Args:
               observable: The operator form of the observable.
               name: The name of the observable.
               matrix_type: The type of matrix used for the observable when
                   creating the matrix wrapper. The default is `sparse`. The
                   other choices are `dense` and `direct`.

           )EOF")
      .def("advance", &ExactTimePropagation::Advance, py::arg("dt"), R"EOF(
           Advance the time propagation by dt.

           Args:
               dt (float): The time step.
      )EOF")
      .def_property("t", &ExactTimePropagation::GetTime,
                    &ExactTimePropagation::SetTime,
                    R"EOF(double: Time in the simulation.)EOF")
      .def_property("state", &ExactTimePropagation::GetState,
                   &ExactTimePropagation::SetState,
                   "Current state.")
      .def(
          "get_observable_stats", &ExactTimePropagation::GetObservableStats,
          R"EOF(
        Calculate and return the value of the operators stored as observables.

        )EOF");
}

}  // namespace netket

#endif
