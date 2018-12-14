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

#define ADDSAMPLERMETHODS(name)                                   \
                                                                  \
  .def("reset", &name::Reset)                                     \
      .def("sweep", &name::Sweep)                                 \
      .def_property("visible", &name::Visible, &name::SetVisible) \
      .def_property_readonly("acceptance", &name::Acceptance)     \
      .def_property_readonly("hilbert", &name::GetHilbert)        \
      .def_property_readonly("machine", &name::GetMachine)

void AddSamplerModule(py::module &m) {
  auto subm = m.def_submodule("sampler");

  py::class_<SamplerType>(subm, "Sampler") ADDSAMPLERMETHODS(SamplerType);

  using SeedableSamplerType = SeedableSampler<MachineType>;
  py::class_<SeedableSamplerType, SamplerType>(subm, "SeedableSampler")
      .def("seed", &SeedableSamplerType::Seed, py::arg("base_seed"));

  {
    using DerSampler = MetropolisLocal<MachineType>;
    py::class_<DerSampler, SeedableSamplerType>(subm, "MetropolisLocal")
        .def(py::init<MachineType &>(), py::keep_alive<1, 2>(),
             py::arg("machine")) ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisLocalPt<MachineType>;
    py::class_<DerSampler, SeedableSamplerType>(subm, "MetropolisLocalPt")
        .def(py::init<MachineType &, int>(), py::keep_alive<1, 2>(),
             py::arg("machine"), py::arg("n_replicas") = 1)
            ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisHop<MachineType>;
    py::class_<DerSampler, SeedableSamplerType>(subm, "MetropolisHop")
        .def(py::init<MachineType &, int>(), py::keep_alive<1, 3>(),
             py::arg("machine"), py::arg("d_max") = 1)
            ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisHamiltonian<MachineType, AbstractOperator>;
    py::class_<DerSampler, SeedableSamplerType>(subm, "MetropolisHamiltonian")
        .def(py::init<MachineType &, AbstractOperator &>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("machine"),
             py::arg("hamiltonian")) ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisHamiltonianPt<MachineType, AbstractOperator>;
    py::class_<DerSampler, SeedableSamplerType>(subm, "MetropolisHamiltonianPt")
        .def(py::init<MachineType &, AbstractOperator &, int>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("machine"),
             py::arg("hamiltonian"), py::arg("n_replicas"))
            ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisExchange<MachineType>;
    py::class_<DerSampler, SeedableSamplerType>(subm, "MetropolisExchange")
        .def(py::init<const AbstractGraph &, MachineType &, int>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("graph"),
             py::arg("machine"), py::arg("d_max") = 1)
            ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = MetropolisExchangePt<MachineType>;
    py::class_<DerSampler, SeedableSamplerType>(subm, "MetropolisExchangePt")
        .def(py::init<const AbstractGraph &, MachineType &, int, int>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("graph"),
             py::arg("machine"), py::arg("d_max") = 1,
             py::arg("n_replicas") = 1) ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = ExactSampler<MachineType>;
    py::class_<DerSampler, SeedableSamplerType>(subm, "ExactSampler")
        .def(py::init<MachineType &>(), py::keep_alive<1, 2>(),
             py::arg("machine")) ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = CustomSampler<MachineType>;
    py::class_<DerSampler, SeedableSamplerType>(subm, "CustomSampler")
        .def(py::init<MachineType &, const LocalOperator &,
                      const std::vector<double> &>(),
             py::keep_alive<1, 2>(), py::arg("machine"),
             py::arg("move_operators"),
             py::arg("move_weights") = std::vector<double>())
            ADDSAMPLERMETHODS(DerSampler);
  }

  {
    using DerSampler = CustomSamplerPt<MachineType>;
    py::class_<DerSampler, SeedableSamplerType>(subm, "CustomSamplerPt")
        .def(py::init<MachineType &, const LocalOperator &,
                      const std::vector<double> &, int>(),
             py::keep_alive<1, 2>(), py::arg("machine"),
             py::arg("move_operators"),
             py::arg("move_weights") = std::vector<double>(),
             py::arg("n_replicas") = 1) ADDSAMPLERMETHODS(DerSampler);
  }
}

}  // namespace netket

#endif
