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

#ifndef NETKET_PYSAMPLER_HPP
#define NETKET_PYSAMPLER_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "Graph/graph.hpp"
#include "Operator/operator.hpp"
#include "Utils/memory_utils.hpp"
#include "Utils/parallel_utils.hpp"
#include "abstract_sampler.hpp"
#include "custom_sampler.hpp"
#include "custom_sampler_pt.hpp"
#include "exact_sampler.hpp"
#include "metropolis_exchange.hpp"
#include "metropolis_exchange_pt.hpp"
#include "metropolis_hamiltonian.hpp"
#include "metropolis_hamiltonian_pt.hpp"
#include "metropolis_hop.hpp"
#include "metropolis_local.hpp"
#include "metropolis_local_pt.hpp"

namespace py = pybind11;

namespace netket {

#define ADDSAMPLERMETHODS(name)              \
                                             \
  .def("reset", &name::Reset)                \
      .def("sweep", &name::Sweep)            \
      .def("get_visible", &name::Visible)    \
      .def("set_visible", &name::SetVisible) \
      .def("acceptance", &name::Acceptance)  \
      .def("get_hilbert", &name::GetHilbert) \
      .def("get_machine", &name::GetMachine)

void AddSamplerModule(py::module &m) {
  auto subm = m.def_submodule("sampler");

  py::class_<AbSamplerType>(subm, "Sampler") ADDSAMPLERMETHODS(AbSamplerType);

  {
    using DerSampler = MetropolisLocal<MachineType>;
    py::class_<DerSampler>(subm, "MetropolisLocal")
        .def(py::init<MachineType>(), py::arg("machine"))
            ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisLocalPt<MachineType>;
    py::class_<DerSampler>(subm, "MetropolisLocalPt")
        .def(py::init<MachineType, int>(), py::arg("machine"),
             py::arg("n_replicas")) ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisHop<MachineType>;
    py::class_<DerSampler>(subm, "MetropolisHop")
        .def(py::init<Graph &, MachineType, int>(), py::arg("graph"),
             py::arg("machine"), py::arg("d_max"))
            ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisHamiltonian<MachineType, AbstractOperator>;
    py::class_<DerSampler>(subm, "MetropolisHamiltonian")
        .def(py::init<MachineType, AbstractOperator &>(), py::arg("machine"),
             py::arg("hamiltonian")) ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisHamiltonianPt<MachineType, AbstractOperator>;
    py::class_<DerSampler>(subm, "MetropolisHamiltonianPt")
        .def(py::init<MachineType, AbstractOperator &, int>(),
             py::arg("machine"), py::arg("hamiltonian"), py::arg("n_replicas"))
            ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisExchange<MachineType>;
    py::class_<DerSampler>(subm, "MetropolisExchange")
        .def(py::init<const Graph &, MachineType, int>(), py::arg("graph"),
             py::arg("machine"), py::arg("d_max") = 1)
            ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisExchangePt<MachineType>;
    py::class_<DerSampler>(subm, "MetropolisExchangePt")
        .def(py::init<const Graph &, MachineType, int, int>(), py::arg("graph"),
             py::arg("machine"), py::arg("d_max") = 1,
             py::arg("n_replicas") = 1) ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = ExactSampler<MachineType>;
    py::class_<DerSampler>(subm, "ExactSampler")
        .def(py::init<MachineType>(), py::arg("machine"))
            ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = CustomSampler<MachineType>;
    py::class_<DerSampler>(subm, "CustomSampler")
        .def(
            py::init<MachineType, const LocalOperator &, std::vector<double>>(),
            py::arg("machine"), py::arg("move_operators"),
            py::arg("move_weights") = std::vector<double>())
            ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = CustomSamplerPt<MachineType>;
    py::class_<DerSampler>(subm, "CustomSamplerPt")
        .def(py::init<MachineType, const LocalOperator &, std::vector<double>,
                      int>(),
             py::arg("machine"), py::arg("move_operators"),
             py::arg("move_weights") = std::vector<double>(),
             py::arg("n_replicas")) ADDSAMPLERMETHODS(DerSampler);
  }
}

}  // namespace netket

#endif
