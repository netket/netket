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

#define ADDLAYERMETHODS(name)                      \
                                                   \
  .def("n_input", &name::Ninput)                   \
      .def("n_output", &name::Noutput)             \
      .def("n_par", &name::Npar)                   \
      .def("get_parameters", &name::GetParameters) \
      .def("set_parameters", &name::SetParameters) \
      .def("init_random_parameters", &name::InitRandomPars);
// TODO add more methods

void AddLayerModule(py::module &m) {
  auto subm = m.def_submodule("layer");

  py::class_<LayerType>(subm, "Layer")
      .def(py::init<FullyConnected<StateType>>())
      .def(py::init<ConvolutionalHypercube<StateType>>())
      .def(py::init<SumOutput<StateType>>())
      .def(py::init<Activation<StateType, Lncosh>>())
      .def(py::init<Activation<StateType, Tanh>>())
      .def(py::init<Activation<StateType, Relu>>()) ADDLAYERMETHODS(LayerType);

  {
    using DerType = FullyConnected<StateType>;
    py::class_<DerType>(subm, "FullyConnected")
        .def(py::init<int, int, bool>(), py::arg("input_size"),
             py::arg("output_size"), py::arg("use_bias") = false)
            ADDLAYERMETHODS(DerType);
    py::implicitly_convertible<DerType, LayerType>();
  }
  {
    using DerType = ConvolutionalHypercube<StateType>;
    py::class_<DerType>(subm, "ConvolutionalHypercube")
        .def(py::init<int, int, int, int, int, int, bool>(), py::arg("length"),
             py::arg("dim"), py::arg("input_channels"),
             py::arg("output_channels"), py::arg("stride") = 1,
             py::arg("kernel_length") = 2, py::arg("use_bias") = false)
            ADDLAYERMETHODS(DerType);
    py::implicitly_convertible<DerType, LayerType>();
  }
  {
    using DerType = SumOutput<StateType>;
    py::class_<DerType>(subm, "SumOutput")
        .def(py::init<int>(), py::arg("input_size")) ADDLAYERMETHODS(DerType);
    py::implicitly_convertible<DerType, LayerType>();
  }
  {
    using DerType = Activation<StateType, Lncosh>;
    py::class_<DerType>(subm, "Lncosh")
        .def(py::init<int>(), py::arg("input_size")) ADDLAYERMETHODS(DerType);
    py::implicitly_convertible<DerType, LayerType>();
  }
  {
    using DerType = Activation<StateType, Tanh>;
    py::class_<DerType>(subm, "Tanh")
        .def(py::init<int>(), py::arg("input_size")) ADDLAYERMETHODS(DerType);
    py::implicitly_convertible<DerType, LayerType>();
  }
  {
    using DerType = Activation<StateType, Relu>;
    py::class_<DerType>(subm, "Relu")
        .def(py::init<int>(), py::arg("input_size")) ADDLAYERMETHODS(DerType);
    py::implicitly_convertible<DerType, LayerType>();
  }
}

}  // namespace netket

#endif
