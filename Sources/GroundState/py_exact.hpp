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

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "exact_diagonalization.hpp"
#include "imaginary_time.hpp"

namespace py = pybind11;

namespace netket {

void AddExactModule(py::module &m) {
  auto m_exact = m.def_submodule("exact");

  py::class_<ImagTimePropagation>(
      m_exact, "ImagTimePropagation",
      R"EOF(Solving for the ground state of the wavefunction using imaginary time propagation.)EOF")
      .def(py::init<const AbstractOperator &, ImagTimePropagation::Stepper &,
                    double, ImagTimePropagation::StateVector,
                    const std::string &>(),
           py::keep_alive<1, 2>(), py::arg("hamiltonian"), py::arg("stepper"),
           py::arg("t0"), py::arg("initial_state"),
           py::arg("matrix_type") = "sparse", R"EOF(
           Constructs an ``ImagTimePropagation`` object from a hamiltonian, a stepper,
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

           Examples:
               Solving 1D ising model with imagniary time propagation.

               ```python
               >>> import netket as nk
               >>> import numpy as np
               >>> L = 20
               >>> graph = nk.graph.Hypercube(L, n_dim=1, pbc=True)
               >>> hilbert = nk.hilbert.Spin(graph, 0.5)
               >>> n_states = hilbert.n_states
               >>> hamiltonian = nk.operator.Ising(hilbert, h=1.0)
               >>> stepper = nk.dynamics.create_timestepper(n_states, rel_tol=1e-10, abs_tol=1e-10)
               >>> output = nk.output.JsonOutputWriter('test.log', 'test.wf')
               >>> psi0 = np.random.rand(n_states)
               >>> driver = nk.exact.ImagTimePropagation(hamiltonian, stepper, t0=0, initial_state=psi0)
               >>> driver.add_observable(hamiltonian, 'Hamiltonian')
               >>> for step in driver.iter(dt=0.05, n_iter=2):
               ...     obs = driver.get_observable_stats()

               ```
           )EOF")
      .def("add_observable", &ImagTimePropagation::AddObservable,
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
      .def("advance", &ImagTimePropagation::Advance, py::arg("dt"), R"EOF(
           Advance the time propagation by dt.

           Args:
               dt (float): The time step.
      )EOF")
      .def_property("t", &ImagTimePropagation::GetTime,
                    &ImagTimePropagation::SetTime,
                    R"EOF(double: Time in the simulation.)EOF")
      .def(
          "get_observable_stats",
          [](const ImagTimePropagation &self) {
            py::dict data;
            self.GetObsManager().InsertAllStats(data);
            return data;
          },
          R"EOF(
        Calculate and return the value of the operators stored as observables.

        )EOF");

  py::class_<eddetail::result_t>(
      m_exact, "EdResult",
      R"EOF(Exact diagonalization of the system hamiltonian using either Lanczos or full diagonalization.)EOF")
      .def_property_readonly(
          "eigenvalues", &eddetail::result_t::eigenvalues,
          R"EOF(vector<double>: The eigenvalues of the hamiltonian of the system.)EOF")
      .def_property_readonly(
          "eigenvectors", &eddetail::result_t::eigenvectors,
          R"EOF(vector<Eigen::Matrix<Complex, Eigen::Dynamic, 1>>: The complex eigenvectors of the system hamiltonian.)EOF")
      .def(
          "mean",
          [](eddetail::result_t &self, AbstractOperator &op, int which) {
            if (which < 0 ||
                static_cast<std::size_t>(which) >= self.eigenvectors().size()) {
              throw InvalidInputError("Invalid eigenvector index `which`");
            }
            return self.mean(op, which);
          },
          py::arg("operator"), py::arg("which") = 0);

  m_exact.def("lanczos_ed", &lanczos_ed, py::arg("operator"),
              py::arg("matrix_free") = false, py::arg("first_n") = 1,
              py::arg("max_iter") = 1000, py::arg("seed") = 42,
              py::arg("precision") = 1.0e-14,
              py::arg("compute_eigenvectors") = false, R"EOF(
              Use the Lanczos algorithm to diagonalize the operator using
              routines from IETL.

              Args:
                  operator: The operator to diagnolize.
                  matrix_free: Indicate whether the operator is stored
                      (sparse/dense) or not (direct). The default is `False`.
                  first_n: The numver of eigenvalues to converge. The default is
                      1.
                  max_iter: The maximum number of iterations. The default is 1000.
                  seed: The random number generator seed. The default is 42.
                  precision: The precision to which the eigenvalues will be
                      computed. The default is 1e-14.
                  comput_eigenvectors: Whether or not to compute the
                      eigenvectors of the operator.  The default is `False`.


              Examples:
                  Testing the numer of eigenvalues saved when solving a simple
                  1D Ising problem.

                  ```python
                  >>> import netket as nk
                  >>> first_n=3
                  >>> g = nk.graph.Hypercube(length=8, n_dim=1, pbc=True)
                  >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
                  >>> ha = nk.operator.Ising(h=1.0, hilbert=hi)
                  >>> res = nk.exact.lanczos_ed(ha, first_n=first_n, compute_eigenvectors=True)
                  >>> print(len(res.eigenvalues) == first_n)
                  True

                  ```

              )EOF");

  m_exact.def("full_ed", &full_ed, py::arg("operator"), py::arg("first_n") = 1,
              py::arg("compute_eigenvectors") = false, R"EOF(
              Diagonalize the operator using routines from IETL.

              Args:
                  operator: The operator to diagnolize.
                  first_n: The numver of eigenvalues to converge. The default is
                      1.
                  comput_eigenvectors: Whether or not to compute the
                      eigenvectors of the operator.  The default is `False`.


              Examples:
                  Testing the numer of eigenvalues saved when solving a simple
                  1D Ising problem.

                  ```python
                  >>> import netket as nk
                  >>> first_n=3
                  >>> g = nk.graph.Hypercube(length=8, n_dim=1, pbc=True)
                  >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
                  >>> ha = nk.operator.Ising(h=1.0, hilbert=hi)
                  >>> res = nk.exact.full_ed(ha, first_n=first_n, compute_eigenvectors=True)
                  >>> print(len(res.eigenvalues) == first_n)
                  True

                  ```

              )EOF");
}

}  // namespace netket

#endif
