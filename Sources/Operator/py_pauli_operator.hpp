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

#ifndef NETKET_PYPAULIOPERATOR_HPP
#define NETKET_PYPAULIOPERATOR_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "bosonhubbard.hpp"

namespace py = pybind11;

namespace netket {

void AddPauliOperator(py::module &subm) {
  py::class_<PauliOperator, AbstractOperator>(
      subm, "PauliOperator",
      R"EOF(A Hamiltonian consisiting of Pauli operators.)EOF")
      .def(py::init([](std::shared_ptr<const AbstractHilbert> hi,
                       std::vector<std::string> ops,
                       std::vector<std::complex<double>> opweights,
                       double cutoff) {
             return PauliOperator{hi, std::move(ops), std::move(opweights),
                                  cutoff};
           }),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("operators"),
           py::arg("weights"), py::arg("cutoff") = 1e-10,
           R"EOF(
           Constructs a new ``PauliOperator`` given a hilbert space, and a
           set of Pauli operators.

           Args:
               hilbert: Hilbert space the operator acts on.
               operators: A list of Pauli operators, e.g. ['IXX', 'XZI'].
               weights: A list of amplitudes of the corresponding Pauli operator.
               cutoff: a cutoff to remove small matrix elements

           Examples:
              Constructs a new ``PauliOperator`` operator.

              ```python
              >>> import netket as nk
              >>> g = nk.graph.Hypercube(length=2, n_dim=1, pbc=False)
              >>> hi = nk.hilbert.Qubit(graph=g)
              >>> op = nk.operator.PauliOperator(hilbert=hi, operators=['XX','ZZ'], weights=[1,1])

              ```
         )EOF");
}

}  // namespace netket

#endif
