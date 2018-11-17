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

#ifndef NETKET_PYLAYER_HPP
#define NETKET_PYLAYER_HPP

#include <mpi.h>
#include "layer.hpp"

namespace py = pybind11;

namespace netket {

#define ADDLAYERMETHODS(name)                     \
                                                  \
  .def("Ninput", &name::Ninput)                   \
      .def("Noutput", &name::Noutput)             \
      .def("Npar", &name::Npar)                   \
      .def("GetParameters", &name::GetParameters) \
      .def("SetParameters", &name::SetParameters) \
      .def("InitRandomPars", &name::InitRandomPars);
// TODO add more methods

void AddLayerModule(py::module &m) {
  auto subm = m.def_submodule("layer");

  py::class_<AbLayerType, std::shared_ptr<AbLayerType>>(subm, "Layer")
      ADDLAYERMETHODS(AbLayerType);

  {
    using LayerType = FullyConnected<MachineType>;
    py::class_<LayerType, AbLayerType, std::shared_ptr<LayerType>>(
        subm, "FullyConnected")
        .def(py::init<AbstractActivation &, int, int, bool>(),
             py::arg("activation"), py::arg("input_size"),
             py::arg("output_size"), py::arg("use_bias") = false)
            ADDLAYERMETHODS(LayerType);
  }
  {
    using LayerType = Convolutional<MachineType>;
    py::class_<LayerType, AbLayerType, std::shared_ptr<LayerType>>(
        subm, "Convolutional")
        .def(py::init<const AbstractGraph &, AbstractActivation &, int, int,
                      int, bool>(),
             py::arg("graph"), py::arg("activation"), py::arg("input_channels"),
             py::arg("output_channels"), py::arg("distance") = 1,
             py::arg("use_bias") = false) ADDLAYERMETHODS(LayerType);
  }
  {
    using LayerType = SumOutput<MachineType>;
    py::class_<LayerType, AbLayerType, std::shared_ptr<LayerType>>(subm,
                                                                   "SumOutput")
        .def(py::init<int>(), py::arg("input_size")) ADDLAYERMETHODS(LayerType);
  }
}

}  // namespace netket

#endif
