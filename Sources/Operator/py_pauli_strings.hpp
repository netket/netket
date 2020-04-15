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

#ifndef NETKET_PYPAULISTRINGS_HPP
#define NETKET_PYPAULISTRINGS_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "pauli_strings.hpp"

namespace py = pybind11;

namespace netket {

void AddPauliStrings(py::module &subm) {
  py::class_<PauliStrings, AbstractOperator, std::shared_ptr<PauliStrings>>(
      subm, "PauliStrings",
      R"EOF(A Hamiltonian consisiting of a product of Pauli operators.)EOF")
      .def(py::init([](std::vector<std::string> ops,
                       std::vector<std::complex<double>> opweights,
                       double cutoff) {
             return PauliStrings{std::move(ops), std::move(opweights), cutoff};
           }),
           py::arg("operators"), py::arg("weights"), py::arg("cutoff") = 1e-10,
           R"EOF(
           Constructs a new ``PauliStrings`` operator given a set of Pauli operators.

           Args:
               operators: A list of Pauli operators in string format, e.g. ['IXX', 'XZI'].
               weights: A list of amplitudes of the corresponding Pauli operator.
               cutoff: a cutoff to remove small matrix elements

           Examples:
               Constructs a new ``PauliOperator`` operator.

               >>> import netket as nk
               >>> op = nk.operator.PauliStrings(operators=['XX','ZZ'], weights=[1,1])
               >>> op.hilbert.size
               2

         )EOF");
}

}  // namespace netket

#endif
