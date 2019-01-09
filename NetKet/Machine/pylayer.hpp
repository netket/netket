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

#define ADDLAYERMETHODS(name)                                                                      \
                                                                                                   \
  .def_property_readonly("n_input", &name::Ninput,                                                 \
                         R"EOF(int: The number of inputs into the layer.)EOF")                     \
      .def_property_readonly(                                                                      \
          "n_output", &name::Noutput,                                                              \
          R"EOF(int: The number of outputs from the layer.)EOF")                                   \
      .def_property_readonly(                                                                      \
          "n_par", &name::Npar,                                                                    \
          R"EOF(int: The number parameters within the layer.)EOF")                                 \
      .def_property(                                                                               \
          "parameters", &name::GetParameters, &name::SetParameters,                                \
          R"EOF(list: List containing the parameters within the layer. Readable and writable)EOF") \
      .def("init_random_parameters", &name::InitRandomPars);
// TODO add more methods

void AddLayerModule(py::module &m) {
  auto subm = m.def_submodule("layer");

  py::class_<LayerType>(subm, "Layer") ADDLAYERMETHODS(LayerType);

  {
    using DerType = FullyConnected<StateType>;
    py::class_<DerType, LayerType>(subm, "FullyConnected", R"EOF(
             A fully connected feedforward layer.)EOF")
        .def(py::init<int, int, bool>(), py::arg("input_size"),
             py::arg("output_size"), py::arg("use_bias") = false, R"EOF(
             Constructs a new ``FullyConnected`` layer given input and output
             sizes.

             Args:
                 input_size: Size of input to the layer (Length of input vector).
                 output_size: Size of output from the layer (Length of output
                              vector).
                 use_bias: If ``True`` then the transformation will include a
                           bias, i.e., the transformation would be affine.

             Examples:
                 A ``FullyConnected`` layer which takes 10-dimensional inputs
                 and gives a 20-dimensional output:

                 ```python
                 >>> from netket.layer import FullyConnected
                 >>> l=FullyConnected(input_size=10,output_size=20,use_bias=True)
                 ```
             )EOF") ADDLAYERMETHODS(DerType);
  }
  {
    using DerType = ConvolutionalHypercube<StateType>;
    py::class_<DerType, LayerType>(subm, "ConvolutionalHypercube", R"EOF(
             A convolutional feedforward layer for hypercubes.)EOF")
        .def(py::init<int, int, int, int, int, int, bool>(), py::arg("length"),
             py::arg("n_dim"), py::arg("input_channels"),
             py::arg("output_channels"), py::arg("stride") = 1,
             py::arg("kernel_length") = 2, py::arg("use_bias") = false, R"EOF(
             Constructs a new ``ConvolutionalHypercube`` layer.

             Args:
                 length: Size of input images.
                 n_dim: Dimension of the input images.
                 input_channels: Number of input channnels.
                 output_channels: Number of output channnels.
                 stride: Stride distance.
                 kernel_length:  Size of the kernels.
                 use_bias: If ``True`` then the transformation will include a
                           bias, i.e., the transformation would be affine.

             Examples:
                 A ``ConvolutionalHypercube`` layer which takes 4 10x10 input images
                 and gives 8 10x10 output images by convolving with 4x4 kernels:

                 ```python
                 >>> from netket.layer import ConvolutionalHypercube
                 >>> l=ConvolutionalHypercube(
                     length=10,
                     n_dim=2,
                     input_channels=4,
                     output_channels=8,
                     kernel_length=4)
                 ```
             )EOF") ADDLAYERMETHODS(DerType);
  }
  {
    using DerType = SumOutput<StateType>;
    py::class_<DerType, LayerType>(subm, "SumOutput", R"EOF(
             A feedforward layer which sums the inputs to give a single output.)EOF")
        .def(py::init<int>(), py::arg("input_size"), R"EOF(
        Constructs a new ``SumOutput`` layer.

        Args:
            input_size: Size of input.

        Examples:
            A ``SumOutput`` layer which takes 10-dimensional inputs:

            ```python
            >>> from netket.layer import SumOutput
            >>> l=SumOutput(input_size=10)
            ```
        )EOF") ADDLAYERMETHODS(DerType);
  }
  {
    using DerType = Activation<StateType, Lncosh>;
    py::class_<DerType, LayerType>(subm, "Lncosh", R"EOF(
             An activation layer which applies Lncosh to each input.)EOF")
        .def(py::init<int>(), py::arg("input_size"), R"EOF(
        Constructs a new ``Lncosh`` activation layer.

        Args:
            input_size: Size of input.

        Examples:
            A ``Lncosh`` activation layer which applies the Lncosh function
            coefficient-wise to a 10-dimensional input:

            ```python
            >>> from netket.layer import Lncosh
            >>> l=Lncosh(input_size=10)
            ```
        )EOF") ADDLAYERMETHODS(DerType);
  }
  {
    using DerType = Activation<StateType, Tanh>;
    py::class_<DerType, LayerType>(subm, "Tanh", R"EOF(
             An activation layer which applies Tanh to each input.)EOF")
        .def(py::init<int>(), py::arg("input_size"), R"EOF(
        Constructs a new ``Tanh`` activation layer.

        Args:
            input_size: Size of input.

        Examples:
            A ``Tanh`` activation layer which applies the Tanh function
            coefficient-wise to a 10-dimensional input:

            ```python
            >>> from netket.layer import Tanh
            >>> l=Tanh(input_size=10)
            ```
        )EOF") ADDLAYERMETHODS(DerType);
  }
  {
    using DerType = Activation<StateType, Relu>;
    py::class_<DerType, LayerType>(subm, "Relu", R"EOF(
             An activation layer which applies ReLu to each input.)EOF")
        .def(py::init<int>(), py::arg("input_size"), R"EOF(
        Constructs a new ``Relu`` activation layer.

        Args:
            input_size: Size of input.

        Examples:
            A ``Relu`` activation layer which applies the Relu function
            coefficient-wise to a 10-dimensional input:

            ```python
            >>> from netket.layer import Relu
            >>> l=Relu(input_size=10)
            ```
        )EOF") ADDLAYERMETHODS(DerType);
  }
}

}  // namespace netket

#endif
