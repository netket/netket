// Copyright 2018 The Simons Foundation, Inc. - All
// Rights Reserved.
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

#ifndef NETKET_PY_NDM_SPIN_PHASE_HPP
#define NETKET_PY_NDM_SPIN_PHASE_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "ndm_spin_phase.hpp"

namespace py = pybind11;

namespace netket {

void AddNdmSpinPhase(py::module &subm) {
  py::class_<NdmSpinPhase, AbstractDensityMatrix>(subm, "NdmSpinPhase", R"EOF(
          A positive semidefinite Neural Density Matrix (NDM) with real-valued parameters.
          In this case, two NDMs are taken to parameterize, respectively, phase
          and amplitude of the density matrix.
          This type of NDM has spin 1/2 hidden and ancilla units.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, int, int, int, int,
                    bool, bool, bool>(),
           py::arg("hilbert"), py::arg("n_hidden") = 0,
           py::arg("n_ancilla") = 0, py::arg("alpha") = 0, py::arg("beta") = 0,
           py::arg("use_visible_bias") = true,
           py::arg("use_hidden_bias") = true,
           py::arg("use_ancilla_bias") = true,
           R"EOF(
                   Constructs a new ``NdmSpinPhase`` machine:

                   Args:
                       hilbert: physical Hilbert space over which the density matrix acts.
                       n_hidden:  Number of hidden units.
                       n_ancilla: Number of ancilla units.
                       alpha: Hidden unit density.
                       beta:  Ancilla unit density.
                       use_visible_bias: If ``True`` then there would be a
                                        bias on the visible units.
                                        Default ``True``.
                       use_hidden_bias:  If ``True`` then there would be a
                                       bias on the visible units.
                                       Default ``True``.
                       use_ancilla_bias: If ``True`` then there would be a
                                       bias on the ancilla units.
                                       Default ``True``.

                   Examples:
                       A ``NdmSpinPhase`` machine with hidden unit density
                       alpha = 1 and ancilla unit density beta = 2 for a
                       one-dimensional L=9 spin-half system:

                       ```python
                       >>> from netket.machine import NdmSpinPhase
                       >>> from netket.hilbert import Spin
                       >>> from netket.graph import Hypercube
                       >>> g = Hypercube(length=9, n_dim=1)
                       >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                       >>> ma = NdmSpinPhase(hilbert=hi,alpha=1, beta=2)
                       >>> print(ma.n_par)
                       1720

                       ```
                   )EOF");
}

}  // namespace netket

#endif  // NETKET_PY_NDM_SPIN_PHASE_HPP
