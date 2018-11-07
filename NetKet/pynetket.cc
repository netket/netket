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

#ifndef NETKET_PYNETKET_CC
#define NETKET_PYNETKET_CC

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "netket.hpp"

namespace py = pybind11;

// PYBIND11_MAKE_OPAQUE(std::vector<int>);
// PYBIND11_MAKE_OPAQUE(std::vector<double>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::complex<double>>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<int>>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<double>>);

namespace netket {

PYBIND11_MODULE(pynetket, m) {
  // py::bind_vector<std::vector<int>>(m, "VectorInt");
  // py::bind_vector<std::vector<std::vector<int>>>(m, "VectorVectorInt");
  // py::bind_vector<std::vector<double>>(m, "VectorDouble");
  // py::bind_vector<std::vector<std::vector<double>>>(m, "VectorVectorDouble");
  // py::bind_vector<std::vector<std::complex<double>>>(m,
  // "VectorComplexDouble");

  // TODO move modules in separate files closer to their binding classes
  py::class_<AbstractGraph>(m, "Graph")
      .def("Nsites", &AbstractGraph::Nsites)
      .def("AdjacencyList", &AbstractGraph::AdjacencyList)
      .def("SymmetryTable", &AbstractGraph::SymmetryTable)
      .def("EdgeColors", &AbstractGraph::EdgeColors)
      .def("IsBipartite", &AbstractGraph::IsBipartite)
      .def("IsConnected", &AbstractGraph::IsConnected)
      .def("Distances", &AbstractGraph::Distances)
      .def("AllDistances", &AbstractGraph::AllDistances);

  py::class_<Hypercube, AbstractGraph>(m, "Hypercube")
      .def(py::init<int, int, bool, std::vector<std::vector<int>>>(),
           py::arg("L"), py::arg("ndim"), py::arg("pbc") = true,
           py::arg("edgecolors") = std::vector<std::vector<int>>())
      .def("Nsites", &Hypercube::Nsites)
      .def("AdjacencyList", &Hypercube::AdjacencyList)
      .def("SymmetryTable", &Hypercube::SymmetryTable)
      .def("EdgeColors", &Hypercube::EdgeColors)
      .def("IsBipartite", &Hypercube::IsBipartite)
      .def("IsConnected", &Hypercube::IsConnected)
      .def("Distances", &Hypercube::Distances)
      .def("AllDistances", &Hypercube::AllDistances);

  py::class_<CustomGraph, AbstractGraph>(m, "CustomGraph")
      .def(
          py::init<int, std::vector<std::vector<int>>,
                   std::vector<std::vector<int>>, std::vector<std::vector<int>>,
                   std::vector<std::vector<int>>, bool>(),
          py::arg("size") = 0,
          py::arg("adjacency_list") = std::vector<std::vector<int>>(),
          py::arg("edges") = std::vector<std::vector<int>>(),
          py::arg("automorphisms") = std::vector<std::vector<int>>(),
          py::arg("edgecolors") = std::vector<std::vector<int>>(),
          py::arg("is_bipartite") = false)
      .def("Nsites", &CustomGraph::Nsites)
      .def("AdjacencyList", &CustomGraph::AdjacencyList)
      .def("SymmetryTable", &CustomGraph::SymmetryTable)
      .def("EdgeColors", &CustomGraph::EdgeColors)
      .def("IsBipartite", &CustomGraph::IsBipartite)
      .def("IsConnected", &CustomGraph::IsConnected)
      .def("Distances", &CustomGraph::Distances)
      .def("AllDistances", &CustomGraph::AllDistances);

  py::class_<AbstractHilbert>(m, "Hilbert")
      .def("IsDiscrete", &AbstractHilbert::IsDiscrete)
      .def("LocalSize", &AbstractHilbert::LocalSize)
      .def("Size", &AbstractHilbert::Size)
      .def("LocalStates", &AbstractHilbert::LocalStates)
      .def("UpdateConf", &AbstractHilbert::UpdateConf)
      .def("GetGraph", &AbstractHilbert::GetGraph);

  py::class_<Spin, AbstractHilbert>(m, "Spin")
      .def(py::init<const AbstractGraph &, double>(), py::arg("graph"),
           py::arg("S"))
      .def(py::init<const AbstractGraph &, double, double>(), py::arg("graph"),
           py::arg("S"), py::arg("total_sz"))
      .def("IsDiscrete", &Spin::IsDiscrete)
      .def("LocalSize", &Spin::LocalSize)
      .def("Size", &Spin::Size)
      .def("LocalStates", &Spin::LocalStates)
      .def("UpdateConf", &Spin::UpdateConf)
      .def("GetGraph", &Spin::GetGraph);

  py::class_<Qubit, AbstractHilbert>(m, "Qubit")
      .def(py::init<const AbstractGraph &>(), py::arg("graph"))
      .def("IsDiscrete", &Qubit::IsDiscrete)
      .def("LocalSize", &Qubit::LocalSize)
      .def("Size", &Qubit::Size)
      .def("LocalStates", &Qubit::LocalStates)
      .def("UpdateConf", &Qubit::UpdateConf)
      .def("GetGraph", &Qubit::GetGraph);

  py::class_<Boson, AbstractHilbert>(m, "Boson")
      .def(py::init<const AbstractGraph &, int>(), py::arg("graph"),
           py::arg("nmax"))
      .def(py::init<const AbstractGraph &, int, int>(), py::arg("graph"),
           py::arg("nmax"), py::arg("nbosons"))
      .def("IsDiscrete", &Boson::IsDiscrete)
      .def("LocalSize", &Boson::LocalSize)
      .def("Size", &Boson::Size)
      .def("LocalStates", &Boson::LocalStates)
      .def("UpdateConf", &Boson::UpdateConf)
      .def("GetGraph", &Boson::GetGraph);

  py::class_<CustomHilbert, AbstractHilbert>(m, "CustomHilbert")
      .def(py::init<const AbstractGraph &, std::vector<double>>(),
           py::arg("graph"), py::arg("local_states"))
      .def("IsDiscrete", &CustomHilbert::IsDiscrete)
      .def("LocalSize", &CustomHilbert::LocalSize)
      .def("Size", &CustomHilbert::Size)
      .def("LocalStates", &CustomHilbert::LocalStates)
      .def("UpdateConf", &CustomHilbert::UpdateConf)
      .def("GetGraph", &CustomHilbert::GetGraph);

  py::class_<AbstractHamiltonian>(m, "Hamiltonian")
      .def("FindConn", &AbstractHamiltonian::FindConn)
      .def("GetHilbert", &AbstractHamiltonian::GetHilbert);

  py::class_<Ising, AbstractHamiltonian>(m, "Ising")
      .def(py::init<const AbstractHilbert &, double, double>(),
           py::arg("hilbert"), py::arg("h"), py::arg("J") = 1.0)
      .def("FindConn", &Ising::FindConn)
      .def("GetHilbert", &Ising::GetHilbert);

  py::class_<Heisenberg, AbstractHamiltonian>(m, "Heisenberg")
      .def(py::init<const AbstractHilbert &>(), py::arg("hilbert"))
      .def("FindConn", &Heisenberg::FindConn)
      .def("GetHilbert", &Heisenberg::GetHilbert);

  py::class_<GraphHamiltonian, AbstractHamiltonian>(m, "GraphHamiltonian")
      .def(py::init<const AbstractHilbert &, GraphHamiltonian::VecType,
                    GraphHamiltonian::VecType, std::vector<int>>(),
           py::arg("hilbert"), py::arg("siteops") = GraphHamiltonian::VecType(),
           py::arg("bondops") = GraphHamiltonian::VecType(),
           py::arg("bondops_colors") = std::vector<int>())
      .def("FindConn", &GraphHamiltonian::FindConn)
      .def("GetHilbert", &GraphHamiltonian::GetHilbert);

  py::class_<CustomHamiltonian, AbstractHamiltonian>(m, "CustomHamiltonian")
      .def(py::init<const AbstractHilbert &, const CustomHamiltonian::VecType &,
                    const std::vector<std::vector<int>> &>(),
           py::arg("hilbert"), py::arg("operators"), py::arg("acting_on"))
      .def("FindConn", &CustomHamiltonian::FindConn)
      .def("GetHilbert", &CustomHamiltonian::GetHilbert);

  py::class_<BoseHubbard, AbstractHamiltonian>(m, "BoseHubbard")
      .def(py::init<const AbstractHilbert &, double, double, double>(),
           py::arg("hilbert"), py::arg("U"), py::arg("V") = 0.,
           py::arg("mu") = 0.)
      .def("FindConn", &BoseHubbard::FindConn)
      .def("GetHilbert", &BoseHubbard::GetHilbert);

  py::class_<SparseMatrixWrapper<AbstractHamiltonian>>(m, "SparseMatrixWrapper")
      .def(py::init<const AbstractHamiltonian &>(), py::arg("hamiltonian"))
      .def("GetMatrix", &SparseMatrixWrapper<AbstractHamiltonian>::GetMatrix);

  using MachineType = std::complex<double>;
  using AbMachineType = AbstractMachine<MachineType>;
  py::class_<AbMachineType>(m, "Machine")
      .def("Npar", &AbMachineType::Npar)
      .def("GetParameters", &AbMachineType::GetParameters)
      .def("SetParameters", &AbMachineType::SetParameters)
      .def("InitRandomPars", &AbMachineType::InitRandomPars)
      .def("Nvisible", &AbMachineType::Nvisible)
      .def("GetHilbert", &AbMachineType::GetHilbert);

  using RbmSpinType = RbmSpin<MachineType>;
  py::class_<RbmSpinType, AbMachineType>(m, "RbmSpin")
      .def(py::init<const AbstractHilbert &, int, int, bool, bool>(),
           py::arg("hilbert"), py::arg("nhidden") = 0, py::arg("alpha") = 0,
           py::arg("use_visible_bias") = true,
           py::arg("use_hidden_bias") = true)
      .def("Npar", &RbmSpinType::Npar)
      .def("GetParameters", &RbmSpinType::GetParameters)
      .def("SetParameters", &RbmSpinType::SetParameters)
      .def("InitRandomPars", &RbmSpinType::InitRandomPars, py::arg("seed"),
           py::arg("sigma"))
      .def("Nvisible", &RbmSpinType::Nvisible)
      .def("GetHilbert", &RbmSpinType::GetHilbert);
  // TODO add other methods

  using SamplerType = AbstractSampler<AbMachineType>;
  py::class_<AbstractSampler<AbMachineType>>(m, "Sampler")
      .def("Reset", &SamplerType::Reset)
      .def("Sweep", &SamplerType::Sweep)
      .def("Visible", &SamplerType::Visible)
      .def("SetVisible", &SamplerType::SetVisible)
      .def("Psi", &SamplerType::Psi)
      .def("Acceptance", &SamplerType::Acceptance);

  py::class_<MetropolisLocal<AbMachineType>, SamplerType>(m, "MetropolisLocal")
      .def(py::init<AbMachineType &>(), py::arg("machine"))
      .def("Reset", &MetropolisLocal<AbMachineType>::Reset)
      .def("Sweep", &MetropolisLocal<AbMachineType>::Sweep)
      .def("Visible", &MetropolisLocal<AbMachineType>::Visible)
      .def("SetVisible", &MetropolisLocal<AbMachineType>::SetVisible)
      .def("Psi", &MetropolisLocal<AbMachineType>::Psi)
      .def("Acceptance", &MetropolisLocal<AbMachineType>::Acceptance);
}

}  // namespace netket

#endif
