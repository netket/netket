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

#ifndef NETKET_PYRBMSPINSYMM_HPP
#define NETKET_PYRBMSPINSYMM_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "rbm_spin_symm.hpp"

namespace py = pybind11;

namespace netket {

void AddRbmSpinSymm(py::module &subm) {
  py::class_<RbmSpinSymm<StateType>, MachineType>(subm, "RbmSpinSymm", R"EOF(
             A fully connected Restricted Boltzmann Machine with lattice
             symmetries. This type of RBM has spin 1/2 hidden units and is
             defined by:

             $$ \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M
                \cosh \left(\sum_i^N W_{ij} s_i + b_j \right) $$

             for arbitrary local quantum numbers $$ s_i $$. However, the weights
             ($$ W_{ij} $$) and biases ($$ a_i $$, $$ b_i $$) respects the
             specified symmetries of the lattice.)EOF")
      .def(py::init<const AbstractHilbert &, int, bool, bool>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("alpha") = 0,
           py::arg("use_visible_bias") = true,
           py::arg("use_hidden_bias") = true,
           R"EOF(
                   Constructs a new ``RbmSpinSymm`` machine:

                   Args:
                       hilbert: Hilbert space object for the system.
                       alpha: Hidden unit density.
                       use_visible_bias: If ``True`` then there would be a
                                        bias on the visible units.
                                        Default ``True``.
                       use_hidden_bias: If ``True`` then there would be a
                                       bias on the visible units.
                                       Default ``True``.

                   Examples:
                       A ``RbmSpinSymm`` machine with hidden unit density
                       alpha = 2 for a one-dimensional L=20 spin-half system:

                       ```python
                       >>> from netket.machine import RbmSpinSymm
                       >>> from netket.hilbert import Spin
                       >>> from netket.graph import Hypercube
                       >>> g = Hypercube(length=20, n_dim=1)
                       >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                       >>> ma = RbmSpinSymm(hilbert=hi, alpha=2)
                       >>> print(ma.n_par)
                       43

                       ```
                   )EOF");
}

}  // namespace netket

#endif
