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

#ifndef NETKET_PYFFNN_HPP
#define NETKET_PYFFNN_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "ffnn.hpp"

namespace py = pybind11;

namespace netket {

void AddFFNN(py::module &subm) {
  {
    using DerMachine = FFNN<StateType>;
    py::class_<DerMachine, MachineType>(subm, "FFNN", R"EOF(
             A feedforward neural network (FFNN) Machine. This machine is
             constructed by providing a sequence of layers from the ``layer``
             class. Each layer implements a transformation such that the
             information is transformed sequentially as it moves from the input
             nodes through the hidden layers and to the output nodes.)EOF")
        .def(py::init([](AbstractHilbert const &hi, py::tuple tuple) {
               auto layers =
                   py::cast<std::vector<AbstractLayer<StateType> *>>(tuple);
               return DerMachine{hi, std::move(layers)};
             }),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("hilbert"),
             py::arg("layers"),
             R"EOF(
              Constructs a new ``FFNN`` machine:

              Args:
                  hilbert: Hilbert space object for the system.
                  layers: Tuple of layers.

              Examples:
                  A ``FFNN`` machine with 2 layers.
                  for a one-dimensional L=20 spin-half system:

                  ```python
                  >>> from netket.layer import SumOutput
                  >>> from netket.layer import FullyConnected
                  >>> from netket.layer import Lncosh
                  >>> from netket.hilbert import Spin
                  >>> from netket.graph import Hypercube
                  >>> from netket.machine import FFNN
                  >>> g = Hypercube(length=20, n_dim=1)
                  >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                  >>> layers = (FullyConnected(input_size=20,output_size=20,use_bias=True),Lncosh(input_size=20),SumOutput(input_size=20))
                  >>> ma = FFNN(hi, layers)
                  >>> print(ma.n_par)
                  420

                  ```
              )EOF");
  }
}

}  // namespace netket

#endif
