// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#include "Machine/py_torch.hpp"

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "Machine/torch.hpp"

namespace py = pybind11;

namespace netket {

void AddPyTorchMachine(PyObject* raw) {
  auto m = py::module{py::reinterpret_borrow<py::object>(raw)};
  py::class_<PyTorchMachine, AbstractMachine>(m, "Torch",
                                              R"EOF(
    A wrapper which allows one to use a `torch.jit.ScriptModule` as a NetKet machine.

    *Note:* Backprop is not yet supported pending implementation of SR in terms
    of Jacobian vector products.
    )EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, std::string>(),
           R"EOF(
           Wraps a `torch.jit.ScriptModule` into a NetKet machine.

           Args:
               hilbert: Hilbert space on which to define the machine.
               filename: Name of the file where PyTorch machine was saved to
                   using `torch.jit.ScriptModule.save`.
           )EOF",
           py::arg{"hilbert"}, py::arg{"filename"});
}

}  // namespace netket
