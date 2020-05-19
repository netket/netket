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

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "Machine/mps_periodic.hpp"
#include "Machine/py_abstract_machine.hpp"
#include "Utils/pybind_helpers.hpp"

namespace py = pybind11;

namespace netket {

namespace {

void AddMpsPeriodic(py::module subm) {
  py::class_<MPSPeriodic, AbstractMachine>(subm, "MPSPeriodic")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, int, bool, int>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("bond_dim"),
           py::arg("diag") = false, py::arg("symperiod") = -1);
}

void AddAbstractMachine(py::module m) {
  py::class_<AbstractMachine, PyAbstractMachine<>>(m, "Machine")
      .def(py::init<std::shared_ptr<AbstractHilbert const>>(),
           py::arg{"hilbert"})
      .def_property_readonly(
          "n_par", &AbstractMachine::Npar,
          R"EOF(int: The number of parameters in the machine.)EOF")
      .def_property("parameters", &AbstractMachine::GetParameters,
                    &AbstractMachine::SetParameters,
                    R"EOF(list: List containing the parameters within the layer.
            Read and write)EOF")
      .def("init_random_parameters", &AbstractMachine::InitRandomPars,
           py::arg{"sigma"} = 0.1, py::arg{"seed"} = py::none(),
           py::arg{"rand_gen"} = py::none(),
           R"EOF(
             Member function to initialise machine parameters.

             Args:
                 sigma: Standard deviation of normal distribution from which
                        parameters are drawn.
                 seed: The random number generator seed. If not given, rand_gen
                       is considered instead.
                 rand_gen: The random number generator (netket.RandomEngine) to be used.
                           If not given, the global random generator (with its current state)
                           is used.

           )EOF")
      .def("log_val",
           [](AbstractMachine &self, py::array_t<double> x,
              nonstd::optional<py::array_t<Complex>> out) {
             if (x.ndim() == 1) {
               auto input = x.cast<Eigen::Ref<const VectorXd>>();
               return py::cast(self.LogValSingle(input));
             } else if (x.ndim() == 2) {
               auto input = x.cast<Eigen::Ref<const RowMatrix<double>>>();
               return py::cast(self.LogVal(input, any{}));
             } else if (x.ndim() == 3) {
               auto input = Eigen::Map<const RowMatrix<double>>{
                   x.data(), x.shape(0) * x.shape(1), x.shape(2)};
               py::array_t<Complex> result =
                   py::cast(self.LogVal(input, any{}));
               result.resize({x.shape(0), x.shape(1)});
               return py::object(result);
             } else {
               throw InvalidInputError{"Invalid input dimension"};
             }
           },
           py::arg("v"), py::arg("out") = py::none(),
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
      .def("der_log_diff",
           [](AbstractMachine &self, AbstractMachine::VisibleConstType v,
              const std::vector<std::vector<int>> &tochange,
              const std::vector<std::vector<double>> &newconfs) {
             return py::cast(self.LogValDiff(v, tochange, newconfs));
           },
           py::arg("v"), py::arg("tochange"), py::arg("newconfs"),
           R"EOF(
                 Member function to obtain the differences in der_log of machine
                 given an input and a change to the input.

                 Args:
                     v: Input vector to machine.
                     tochange: list containing the indices of the input to be
                         changed
                     newconfs: list containing the new (changed) values at the
                         indices specified in tochange
           )EOF")
      .def("der_log",
           [](AbstractMachine &self, py::array_t<double> x) {
             if (x.ndim() == 1) {
               auto input = x.cast<Eigen::Ref<const VectorXd>>();
               return py::cast(self.DerLogSingle(input));
             } else if (x.ndim() == 2) {
               auto input = x.cast<Eigen::Ref<const RowMatrix<double>>>();
               return py::cast(self.DerLog(input, any{}));
             } else if (x.ndim() == 3) {
               auto input = Eigen::Map<const RowMatrix<double>>{
                   x.data(), x.shape(0) * x.shape(1), x.shape(2)};
               py::array_t<Complex> result =
                   py::cast(self.DerLog(input, any{}));
               result.resize({x.shape(0), x.shape(1),
                              static_cast<pybind11::ssize_t>(self.Npar())});
               return py::object(result);
             } else {
               throw InvalidInputError{"Invalid input dimension"};
             }
           },
           py::arg("v"),
           R"EOF(
                 Member function to obtain the derivatives of log value of
                 machine given an input wrt the machine's parameters.

                 Args:
                     v: Input vector to machine.
           )EOF")
      .def("vector_jacobian_prod", &AbstractMachine::VectorJacobianProd,
           py::arg("v"), py::arg("vec"), py::arg("out"),
           R"EOF(
             Computes the scalar product between gradient of the logarithm of the wavefunction for a
             batch of visible configurations `x` and a vector `vec`. The result is stored into `out`.

             Args:
                  x: a matrix of `float64` of shape `(*, self.n_visible)`.
                  vec: a `complex128` vector used to compute the inner product with the jacobian.
                  out: The result of the inner product, it is a vector of `complex128` and length `self.n_par`.


             Returns:
                  `out`
                )EOF")
      .def_property_readonly(
          "n_visible", &AbstractMachine::Nvisible,
          R"EOF(int: The number of inputs into the machine aka visible units in
            the case of Restricted Boltzmann Machines.)EOF")
      .def_property_readonly(
          "input_size", &AbstractMachine::Nvisible,
          R"EOF(int: The number of inputs into the machine.)EOF")
      .def_property_readonly(
          "hilbert", &AbstractMachine::GetHilbert,
          R"EOF(netket.hilbert.Hilbert: The hilbert space object of the system.)EOF")
      .def_property_readonly(
          "is_holomorphic", &AbstractMachine::IsHolomorphic,
          R"EOF(bool: Whether the given wave-function is a holomorphic function of
            its parameters )EOF")
      .def("save", &AbstractMachine::Save, py::arg("filename"),
           R"EOF(
                 Member function to save the machine parameters.

                 Args:
                     filename: name of file to save parameters to.
           )EOF")
      .def("load", &AbstractMachine::Load, py::arg("filename"),
           R"EOF(
                 Member function to load machine parameters from a json file.

                 Args:
                     filename: name of file to load parameters from.
           )EOF")
      .def(
          "state_dict",
          [](AbstractMachine &self) {
            return py::reinterpret_steal<py::object>(self.StateDict());
          },
          R"EOF(Returns machine's state as a dictionary. Similar to `torch.nn.Module.state_dict`.
           )EOF")
      .def("load_state_dict",
           [](AbstractMachine &self, py::dict state) {
             self.StateDict(state.ptr());
           },
           R"EOF(Loads machine's state from `state`.
           )EOF")
      .def("to_array",
           [](AbstractMachine &self,
              bool normalize) -> AbstractMachine::VectorType {
             const auto &hind = self.GetHilbert().GetIndex();
             AbstractMachine::VectorType vals(hind.NStates());

             double maxlog = std::numeric_limits<double>::lowest();

             for (Index i = 0; i < hind.NStates(); i++) {
               vals(i) = self.LogValSingle(hind.NumberToState(i));
               if (std::real(vals(i)) > maxlog) {
                 maxlog = std::real(vals(i));
               }
             }

             for (Index i = 0; i < hind.NStates(); i++) {
               vals(i) -= maxlog;
               vals(i) = std::exp(vals(i));
             }

             if (normalize) {
               vals.normalize();
             }
             return vals;
           },
           py::arg("normalize") = true,
           R"EOF(
                Returns a numpy array representation of the machine.
                The returned array is normalized to 1 in L2 norm.
                Note that, in general, the size of the array is exponential
                in the number of quantum numbers, and this operation should thus
                only be performed for low-dimensional Hilbert spaces.

                This method requires an indexable Hilbert space.
              )EOF")
      .def("log_norm",
           [](AbstractMachine &self) -> double {
             const auto &hind = self.GetHilbert().GetIndex();
             AbstractMachine::VectorType vals(hind.NStates());

             double maxlog = std::numeric_limits<double>::lowest();

             for (Index i = 0; i < hind.NStates(); i++) {
               vals(i) = self.LogValSingle(hind.NumberToState(i));
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
}

}  // namespace

void AddMachineModule(py::module m) {
  auto subm = m.def_submodule("machine");

  AddAbstractMachine(subm);
  AddMpsPeriodic(subm);
}

}  // namespace netket
