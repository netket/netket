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

#include "Machine/py_machine.hpp"

#include <complex>
#include <limits>
#include <vector>

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "DensityMatrices/py_density_matrix.hpp"
#include "Machine/ffnn.hpp"
#include "Machine/jastrow.hpp"
#include "Machine/jastrow_symm.hpp"
#include "Machine/mps_periodic.hpp"
#include "Machine/rbm_multival.hpp"
#include "Machine/rbm_spin.hpp"
#include "Machine/rbm_spin_phase.hpp"
#include "Machine/rbm_spin_real.hpp"
#include "Machine/rbm_spin_symm.hpp"

namespace py = pybind11;

namespace netket {

namespace {
void AddRbmSpin(py::module subm) {
  py::class_<RbmSpin, AbstractMachine>(subm, "RbmSpin", R"EOF(
          A fully connected Restricted Boltzmann Machine (RBM). This type of
          RBM has spin 1/2 hidden units and is defined by:

          $$ \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M \cosh
             \left(\sum_i^N W_{ij} s_i + b_j \right) $$

          for arbitrary local quantum numbers $$ s_i $$.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, int, int, bool,
                    bool>(),
           py::arg("hilbert"), py::arg("n_hidden") = 0, py::arg("alpha") = 0,
           py::arg("use_visible_bias") = true,
           py::arg("use_hidden_bias") = true,
           R"EOF(
                   Constructs a new ``RbmSpin`` machine:

                   Args:
                       hilbert: Hilbert space object for the system.
                       n_hidden: Number of hidden units.
                       alpha: Hidden unit density.
                       use_visible_bias: If ``True`` then there would be a
                                        bias on the visible units.
                                        Default ``True``.
                       use_hidden_bias: If ``True`` then there would be a
                                       bias on the visible units.
                                       Default ``True``.

                   Examples:
                       A ``RbmSpin`` machine with hidden unit density
                       alpha = 2 for a one-dimensional L=20 spin-half system:

                       ```python
                       >>> from netket.machine import RbmSpin
                       >>> from netket.hilbert import Spin
                       >>> from netket.graph import Hypercube
                       >>> g = Hypercube(length=20, n_dim=1)
                       >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                       >>> ma = RbmSpin(hilbert=hi,alpha=2)
                       >>> print(ma.n_par)
                       860

                       ```
                   )EOF");
}

void AddRbmSpinSymm(py::module subm) {
  py::class_<RbmSpinSymm, AbstractMachine>(subm, "RbmSpinSymm", R"EOF(
             A fully connected Restricted Boltzmann Machine with lattice
             symmetries. This type of RBM has spin 1/2 hidden units and is
             defined by:

             $$ \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M
                \cosh \left(\sum_i^N W_{ij} s_i + b_j \right) $$

             for arbitrary local quantum numbers $$ s_i $$. However, the weights
             ($$ W_{ij} $$) and biases ($$ a_i $$, $$ b_i $$) respects the
             specified symmetries of the lattice.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, int, bool, bool>(),
           py::arg("hilbert"), py::arg("alpha") = 0,
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

void AddRbmMultival(py::module subm) {
  py::class_<RbmMultival, AbstractMachine>(subm, "RbmMultiVal", R"EOF(
             A fully connected Restricted Boltzmann Machine for handling larger
             local Hilbert spaces.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, int, int, bool,
                    bool>(),
           py::arg("hilbert"), py::arg("n_hidden") = 0, py::arg("alpha") = 0,
           py::arg("use_visible_bias") = true,
           py::arg("use_hidden_bias") = true);
}

void AddRbmSpinPhase(py::module subm) {
  py::class_<RbmSpinPhase, AbstractMachine>(subm, "RbmSpinPhase", R"EOF(
          A fully connected Restricted Boltzmann Machine (RBM) with real-valued parameters.
          In this case, two RBMs are taken to parameterize, respectively, phase
          and amplitude of the wave-function.
          This type of RBM has spin 1/2 hidden units and is defined by:

          $$ \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M \cosh
             \left(\sum_i^N W_{ij} s_i + b_j \right) $$

          for arbitrary local quantum numbers $$ s_i $$.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, int, int, bool,
                    bool>(),
           py::arg("hilbert"), py::arg("n_hidden") = 0, py::arg("alpha") = 0,
           py::arg("use_visible_bias") = true,
           py::arg("use_hidden_bias") = true,
           R"EOF(
                   Constructs a new ``RbmSpinPhase`` machine:

                   Args:
                       hilbert: Hilbert space object for the system.
                       n_hidden: Number of hidden units.
                       alpha: Hidden unit density.
                       use_visible_bias: If ``True`` then there would be a
                                        bias on the visible units.
                                        Default ``True``.
                       use_hidden_bias: If ``True`` then there would be a
                                       bias on the visible units.
                                       Default ``True``.

                   Examples:
                       A ``RbmSpinPhase`` machine with hidden unit density
                       alpha = 2 for a one-dimensional L=20 spin-half system:

                       ```python
                       >>> from netket.machine import RbmSpinPhase
                       >>> from netket.hilbert import Spin
                       >>> from netket.graph import Hypercube
                       >>> g = Hypercube(length=20, n_dim=1)
                       >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                       >>> ma = RbmSpinPhase(hilbert=hi,alpha=2)
                       >>> print(ma.n_par)
                       1720

                       ```
                   )EOF");
}

void AddRbmSpinReal(py::module subm) {
  py::class_<RbmSpinReal, AbstractMachine>(subm, "RbmSpinReal", R"EOF(
          A fully connected Restricted Boltzmann Machine (RBM) with real-valued parameters.
          This type of RBM has spin 1/2 hidden units and is defined by:

          $$ \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M \cosh
             \left(\sum_i^N W_{ij} s_i + b_j \right) $$

          for arbitrary local quantum numbers $$ s_i $$.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, int, int, bool,
                    bool>(),
           py::arg("hilbert"), py::arg("n_hidden") = 0, py::arg("alpha") = 0,
           py::arg("use_visible_bias") = true,
           py::arg("use_hidden_bias") = true,
           R"EOF(
                   Constructs a new ``RbmSpinReal`` machine:

                   Args:
                       hilbert: Hilbert space object for the system.
                       n_hidden: Number of hidden units.
                       alpha: Hidden unit density.
                       use_visible_bias: If ``True`` then there would be a
                                        bias on the visible units.
                                        Default ``True``.
                       use_hidden_bias: If ``True`` then there would be a
                                       bias on the visible units.
                                       Default ``True``.

                   Examples:
                       A ``RbmSpinReal`` machine with hidden unit density
                       alpha = 2 for a one-dimensional L=20 spin-half system:

                       ```python
                       >>> from netket.machine import RbmSpinReal
                       >>> from netket.hilbert import Spin
                       >>> from netket.graph import Hypercube
                       >>> g = Hypercube(length=20, n_dim=1)
                       >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                       >>> ma = RbmSpinReal(hilbert=hi,alpha=2)
                       >>> print(ma.n_par)
                       860

                       ```
                   )EOF");
}

void AddFFNN(py::module subm) {
  py::class_<FFNN, AbstractMachine>(subm, "FFNN", R"EOF(
             A feedforward neural network (FFNN) Machine. This machine is
             constructed by providing a sequence of layers from the ``layer``
             class. Each layer implements a transformation such that the
             information is transformed sequentially as it moves from the input
             nodes through the hidden layers and to the output nodes.)EOF")
      .def(py::init(
               [](std::shared_ptr<const AbstractHilbert> hi, py::tuple tuple) {
                 auto layers = py::cast<std::vector<AbstractLayer *>>(tuple);
                 return FFNN{std::move(hi), std::move(layers)};
               }),
           py::keep_alive<1, 3>(), py::arg("hilbert"), py::arg("layers"),
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

void AddJastrow(py::module subm) {
  py::class_<Jastrow, AbstractMachine>(subm, "Jastrow", R"EOF(
           A Jastrow wavefunction Machine. This machine defines the following
           wavefunction:

           $$ \Psi(s_1,\dots s_N) = e^{\sum_{ij} s_i W_{ij} s_i}$$

           where $$ W_{ij} $$ are the Jastrow parameters.
           )EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>>(),
           py::arg("hilbert"), R"EOF(
                 Constructs a new ``Jastrow`` machine:

                 Args:
                     hilbert: Hilbert space object for the system.

                 Examples:
                     A ``Jastrow`` machine for a one-dimensional L=20 spin 1/2
                     system:

                     ```python
                     >>> from netket.machine import Jastrow
                     >>> from netket.hilbert import Spin
                     >>> from netket.graph import Hypercube
                     >>> g = Hypercube(length=20, n_dim=1)
                     >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                     >>> ma = Jastrow(hilbert=hi)
                     >>> print(ma.n_par)
                     190

                     ```
                 )EOF");
}

void AddJastrowSymm(py::module subm) {
  py::class_<JastrowSymm, AbstractMachine>(subm, "JastrowSymm", R"EOF(
           A Jastrow wavefunction Machine with lattice symmetries.This machine
           defines the wavefunction as follows:

           $$ \Psi(s_1,\dots s_N) = e^{\sum_{ij} s_i W_{ij} s_i}$$

           where $$ W_{ij} $$ are the Jastrow parameters respects the
           specified symmetries of the lattice.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>>(),
           py::arg("hilbert"), R"EOF(
                 Constructs a new ``JastrowSymm`` machine:

                 Args:
                     hilbert: Hilbert space object for the system.

                 Examples:
                     A ``JastrowSymm`` machine for a one-dimensional L=20 spin
                     1/2 system:

                     ```python
                     >>> from netket.machine import JastrowSymm
                     >>> from netket.hilbert import Spin
                     >>> from netket.graph import Hypercube
                     >>> g = Hypercube(length=20, n_dim=1)
                     >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                     >>> ma = JastrowSymm(hilbert=hi)
                     >>> print(ma.n_par)
                     10

                     ```
                 )EOF");
}

void AddMpsPeriodic(py::module subm) {
  py::class_<MPSPeriodic, AbstractMachine>(subm, "MPSPeriodic")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, int, bool, int>(),
           py::arg("hilbert"), py::arg("bond_dim"), py::arg("diag") = false,
           py::arg("symperiod") = -1);
}

void AddLayerModule(py::module m) {
  auto subm = m.def_submodule("layer");

  py::class_<AbstractLayer>(subm, "Layer")
      .def_property_readonly(
          "n_input", &AbstractLayer::Ninput,
          R"EOF(int: The number of inputs into the layer.)EOF")
      .def_property_readonly(
          "n_output", &AbstractLayer::Noutput,
          R"EOF(int: The number of outputs from the layer.)EOF")
      .def_property_readonly(
          "n_par", &AbstractLayer::Npar,
          R"EOF(int: The number parameters within the layer.)EOF")
      .def_property("parameters", &AbstractLayer::GetParameters,
                    &AbstractLayer::SetParameters,
                    R"EOF(list: List containing the parameters within the layer.
            Readable and writable)EOF")
      .def("init_random_parameters", &AbstractLayer::InitRandomPars,
           py::arg("seed") = 1234, py::arg("sigma") = 0.1, R"EOF(
        Member function to initialise layer parameters.

        Args:
            seed: The random number generator seed.
            sigma: Standard deviation of normal distribution from which
                parameters are drawn.
      )EOF");
  // TODO add more methods

  {
    using DerType = FullyConnected;
    py::class_<DerType, AbstractLayer>(subm, "FullyConnected", R"EOF(
             A fully connected feedforward layer. This layer implements the
             transformation from a m-dimensional input vector
             $$ \boldsymbol{v}_n $$ to a n-dimensional output vector
             $$ \boldsymbol{v}_{n+1} $$:

             $$ \boldsymbol{v}_n \rightarrow \boldsymbol{v}_{n+1} =
             g_{n}(\boldsymbol{W}{n}\boldsymbol{v}{n} + \boldsymbol{b}_{n} ) $$

             where $$ \boldsymbol{W}{n} $$ is a m by n weights matrix and
             $$ \boldsymbol{b}_{n} $$ is a n-dimensional bias vector.
             )EOF")
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
                 >>> print(l.n_par)
                 220

                 ```
             )EOF");
  }
  {
    using DerType = ConvolutionalHypercube;
    py::class_<DerType, AbstractLayer>(subm, "ConvolutionalHypercube", R"EOF(
             A convolutional feedforward layer for hypercubes. This layer works
             only for the ``Hypercube`` graph defined in ``graph``. This layer
             implements the standard convolution with periodic boundary
             conditions.)EOF")
        .def(py::init<int, int, int, int, int, int, bool>(), py::arg("length"),
             py::arg("n_dim"), py::arg("input_channels"),
             py::arg("output_channels"), py::arg("stride") = 1,
             py::arg("kernel_length") = 2, py::arg("use_bias") = false, R"EOF(
             Constructs a new ``ConvolutionalHypercube`` layer.

             Args:
                 length: Size of input images.
                 n_dim: Dimension of the input images.
                 input_channels: Number of input channels.
                 output_channels: Number of output channels.
                 stride: Stride distance.
                 kernel_length:  Size of the kernels.
                 use_bias: If ``True`` then the transformation will include a
                           bias, i.e., the transformation would be affine.

             Examples:
                 A ``ConvolutionalHypercube`` layer which takes 4 10x10 input images
                 and gives 8 10x10 output images by convolving with 4x4 kernels:

                 ```python
                 >>> from netket.layer import ConvolutionalHypercube
                 >>> l=ConvolutionalHypercube(length=10,n_dim=2,input_channels=4,output_channels=8,kernel_length=4)
                 >>> print(l.n_par)
                 512

                 ```
             )EOF");
  }
  {
    using DerType = SumOutput;
    py::class_<DerType, AbstractLayer>(subm, "SumOutput", R"EOF(
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
            >>> print(l.n_par)
            0

            ```
        )EOF");
  }
  {
    using DerType = Activation<Lncosh>;
    py::class_<DerType, AbstractLayer>(subm, "Lncosh", R"EOF(
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
            >>> print(l.n_par)
            0

            ```
        )EOF");
  }
  {
    using DerType = Activation<Tanh>;
    py::class_<DerType, AbstractLayer>(subm, "Tanh", R"EOF(
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
            >>> print(l.n_par)
            0

            ```
        )EOF");
  }
  {
    using DerType = Activation<Relu>;
    py::class_<DerType, AbstractLayer>(subm, "Relu", R"EOF(
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
            >>> print(l.n_par)
            0

            ```
        )EOF");
  }
}

}  // namespace

void AddMachineModule(py::module m) {
  auto subm = m.def_submodule("machine");

  py::class_<AbstractMachine>(subm, "Machine")
      .def_property_readonly(
          "n_par", &AbstractMachine::Npar,
          R"EOF(int: The number of parameters in the machine.)EOF")
      .def_property("parameters", &AbstractMachine::GetParameters,
                    &AbstractMachine::SetParameters,
                    R"EOF(list: List containing the parameters within the layer.
            Read and write)EOF")
      .def("init_random_parameters", &AbstractMachine::InitRandomPars,
           py::arg("seed") = 1234, py::arg("sigma") = 0.1,
           R"EOF(
             Member function to initialise machine parameters.

             Args:
                 seed: The random number generator seed.
                 sigma: Standard deviation of normal distribution from which
                     parameters are drawn.
           )EOF")
      .def("log_val",
           (Complex(AbstractMachine::*)(AbstractMachine::VisibleConstType)) &
               AbstractMachine::LogVal,
           py::arg("v"),
           R"EOF(
                 Member function to obtain log value of machine given an input
                 vector.

                 Args:
                     v: Input vector to machine.
           )EOF")
      .def("log_val_diff",
           (AbstractMachine::VectorType(AbstractMachine::*)(
               AbstractMachine::VisibleConstType,
               const std::vector<std::vector<int>> &,
               const std::vector<std::vector<double>> &)) &
               AbstractMachine::LogValDiff,
           py::arg("v"), py::arg("tochange"), py::arg("newconf"),
           R"EOF(
                 Member function to obtain difference in log value of machine
                 given an input and a change to the input.

                 Args:
                     v: Input vector to machine.
                     tochange: list containing the indices of the input to be
                         changed
                     newconf: list containing the new (changed) values at the
                         indices specified in tochange
           )EOF")
      .def("der_log",
           (AbstractMachine::VectorType(AbstractMachine::*)(
               AbstractMachine::VisibleConstType)) &
               AbstractMachine::DerLog,
           py::arg("v"),
           R"EOF(
                 Member function to obtain the derivatives of log value of
                 machine given an input wrt the machine's parameters.

                 Args:
                     v: Input vector to machine.
           )EOF")
      .def_property_readonly(
          "n_visible", &AbstractMachine::Nvisible,
          R"EOF(int: The number of inputs into the machine aka visible units in
            the case of Restricted Boltzmann Machines.)EOF")
      .def_property_readonly(
          "hilbert", &AbstractMachine::GetHilbert,
          R"EOF(netket.hilbert.Hilbert: The hilbert space object of the system.)EOF")
      .def_property_readonly(
          "is_holomorphic", &AbstractMachine::IsHolomorphic,
          R"EOF(bool: Whether the given wave-function is a holomorphic function of
            its parameters )EOF")
      .def(
          "save",
          [](const AbstractMachine &a, std::string filename) {
            json j;
            a.to_json(j);
            std::ofstream filewf(filename);
            filewf << j << std::endl;
            filewf.close();
          },
          py::arg("filename"),
          R"EOF(
                 Member function to save the machine parameters.

                 Args:
                     filename: name of file to save parameters to.
           )EOF")
      .def(
          "load",
          [](AbstractMachine &a, std::string filename) {
            std::ifstream filewf(filename);
            if (filewf.is_open()) {
              json j;
              filewf >> j;
              filewf.close();
              a.from_json(j);
            }
          },
          py::arg("filename"),
          R"EOF(
                 Member function to load machine parameters from a json file.

                 Args:
                     filename: name of file to load parameters from.
           )EOF")
      .def(
          "to_array",
          [](AbstractMachine &self) -> AbstractMachine::VectorType {
            const auto &hind = self.GetHilbert().GetIndex();
            AbstractMachine::VectorType vals(hind.NStates());

            double maxlog = std::numeric_limits<double>::lowest();

            for (Index i = 0; i < hind.NStates(); i++) {
              vals(i) = self.LogVal(hind.NumberToState(i));
              if (std::real(vals(i)) > maxlog) {
                maxlog = std::real(vals(i));
              }
            }

            for (Index i = 0; i < hind.NStates(); i++) {
              vals(i) -= maxlog;
              vals(i) = std::exp(vals(i));
            }

            vals /= vals.norm();
            return vals;
          },
          R"EOF(
                Returns a numpy array representation of the machine.
                The returned array is normalized to 1 in L2 norm.
                Note that, in general, the size of the array is exponential
                in the number of quantum numbers, and this operation should thus
                only be performed for low-dimensional Hilbert spaces.

                This method requires an indexable Hilbert space.
              )EOF")
      .def(
          "log_norm",
          [](AbstractMachine &self) -> double {
            const auto &hind = self.GetHilbert().GetIndex();
            AbstractMachine::VectorType vals(hind.NStates());

            double maxlog = std::numeric_limits<double>::lowest();

            for (Index i = 0; i < hind.NStates(); i++) {
              vals(i) = self.LogVal(hind.NumberToState(i));
              if (std::real(vals(i)) > maxlog) {
                maxlog = std::real(vals(i));
              }
            }

            double norpsi = 0;
            for (Index i = 0; i < hind.NStates(); i++) {
              vals(i) -= maxlog;
              norpsi += std::norm(std::exp(vals(i)));
            }

            return std::log(norpsi) + 2. * maxlog;
          },
          R"EOF(
                Returns the log of the L2 norm of the wave-function.
                This operation is a brute-force calculation, and should thus
                only be performed for low-dimensional Hilbert spaces.

                This method requires an indexable Hilbert space.
                )EOF");

  AddRbmSpin(subm);
  AddRbmSpinSymm(subm);
  AddRbmMultival(subm);
  AddRbmSpinReal(subm);
  AddRbmSpinPhase(subm);
  AddJastrow(subm);
  AddJastrowSymm(subm);
  AddMpsPeriodic(subm);
  AddFFNN(subm);
  AddLayerModule(m);

  AddDensityMatrixModule(subm);
}

}  // namespace netket
