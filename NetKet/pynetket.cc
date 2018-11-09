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

  using AbActivationType = AbstractActivation;
  py::class_<AbActivationType>(m, "Activation")
      .def("__call__", &AbActivationType::operator())
      .def("ApplyJacobian", &AbActivationType::ApplyJacobian);

  {
    using WfType = RbmSpin<MachineType>;
    py::class_<WfType, AbMachineType>(m, "RbmSpin")
        .def(py::init<const AbstractHilbert &, int, int, bool, bool>(),
             py::arg("hilbert"), py::arg("nhidden") = 0, py::arg("alpha") = 0,
             py::arg("use_visible_bias") = true,
             py::arg("use_hidden_bias") = true)
        .def("Npar", &WfType::Npar)
        .def("GetParameters", &WfType::GetParameters)
        .def("SetParameters", &WfType::SetParameters)
        .def("InitRandomPars", &WfType::InitRandomPars, py::arg("seed"),
             py::arg("sigma"))
        .def("Nvisible", &WfType::Nvisible)
        .def("GetHilbert", &WfType::GetHilbert);
    // TODO add other methods?
  }

  {
    using WfType = RbmSpinSymm<MachineType>;
    py::class_<WfType, AbMachineType>(m, "RbmSpinSymm")
        .def(py::init<const AbstractHilbert &, int, int, bool, bool>(),
             py::arg("hilbert"), py::arg("nhidden") = 0, py::arg("alpha") = 0,
             py::arg("use_visible_bias") = true,
             py::arg("use_hidden_bias") = true)
        .def("Npar", &WfType::Npar)
        .def("GetParameters", &WfType::GetParameters)
        .def("SetParameters", &WfType::SetParameters)
        .def("InitRandomPars", &WfType::InitRandomPars, py::arg("seed"),
             py::arg("sigma"))
        .def("Nvisible", &WfType::Nvisible)
        .def("GetHilbert", &WfType::GetHilbert);
    // TODO add other methods?
  }

  {
    using WfType = RbmMultival<MachineType>;
    py::class_<WfType, AbMachineType>(m, "RbmMultival")
        .def(py::init<const AbstractHilbert &, int, int, bool, bool>(),
             py::arg("hilbert"), py::arg("nhidden") = 0, py::arg("alpha") = 0,
             py::arg("use_visible_bias") = true,
             py::arg("use_hidden_bias") = true)
        .def("Npar", &WfType::Npar)
        .def("GetParameters", &WfType::GetParameters)
        .def("SetParameters", &WfType::SetParameters)
        .def("InitRandomPars", &WfType::InitRandomPars, py::arg("seed"),
             py::arg("sigma"))
        .def("Nvisible", &WfType::Nvisible)
        .def("GetHilbert", &WfType::GetHilbert);
    // TODO add other methods?
  }
  {
    using WfType = Jastrow<MachineType>;
    py::class_<WfType, AbMachineType>(m, "Jastrow")
        .def(py::init<const AbstractHilbert &>(), py::arg("hilbert"))
        .def("Npar", &WfType::Npar)
        .def("GetParameters", &WfType::GetParameters)
        .def("SetParameters", &WfType::SetParameters)
        .def("InitRandomPars", &WfType::InitRandomPars, py::arg("seed"),
             py::arg("sigma"))
        .def("Nvisible", &WfType::Nvisible)
        .def("GetHilbert", &WfType::GetHilbert);
    // TODO add other methods?
  }
  {
    using WfType = JastrowSymm<MachineType>;
    py::class_<WfType, AbMachineType>(m, "JastrowSymm")
        .def(py::init<const AbstractHilbert &>(), py::arg("hilbert"))
        .def("Npar", &WfType::Npar)
        .def("GetParameters", &WfType::GetParameters)
        .def("SetParameters", &WfType::SetParameters)
        .def("InitRandomPars", &WfType::InitRandomPars, py::arg("seed"),
             py::arg("sigma"))
        .def("Nvisible", &WfType::Nvisible)
        .def("GetHilbert", &WfType::GetHilbert);
    // TODO add other methods?
  }

  // FEED-FORWARD NETWORK RELATED BINDINGS
  // ACTIVATION FUNCTIONS
  // TODO maybe move these into a separate python modules
  {
    using ActivationType = Tanh;
    py::class_<ActivationType, AbActivationType>(m, "Tanh")
        .def(py::init<>())
        .def("__call__", &ActivationType::operator())
        .def("ApplyJacobian", &ActivationType::ApplyJacobian);
  }
  {
    using ActivationType = Identity;
    py::class_<ActivationType, AbActivationType>(m, "Identity")
        .def(py::init<>())
        .def("__call__", &ActivationType::operator())
        .def("ApplyJacobian", &ActivationType::ApplyJacobian);
  }
  {
    using ActivationType = Lncosh;
    py::class_<ActivationType, AbActivationType>(m, "Lncosh")
        .def(py::init<>())
        .def("__call__", &ActivationType::operator())
        .def("ApplyJacobian", &ActivationType::ApplyJacobian);
  }

  // LAYERS
  // TODO maybe move these into a separate python modules
  using AbLayerType = AbstractLayer<MachineType>;
  {
    py::class_<AbLayerType, std::shared_ptr<AbLayerType>>(m, "Layer")
        .def("Ninput", &AbLayerType::Ninput)
        .def("Noutput", &AbLayerType::Noutput)
        .def("Npar", &AbLayerType::Npar)
        .def("GetParameters", &AbLayerType::GetParameters)
        .def("SetParameters", &AbLayerType::SetParameters)
        .def("InitRandomPars", &AbLayerType::InitRandomPars);
    // TODO add more methods
  }
  {
    using LayerType = FullyConnected<MachineType>;
    py::class_<LayerType, AbLayerType, std::shared_ptr<LayerType>>(
        m, "FullyConnected")
        .def(py::init<AbActivationType &, int, int, bool>(),
             py::arg("activation"), py::arg("input_size"),
             py::arg("output_size"), py::arg("use_bias") = false)
        .def("Ninput", &LayerType::Ninput)
        .def("Noutput", &LayerType::Noutput)
        .def("Npar", &LayerType::Npar)
        .def("GetParameters", &LayerType::GetParameters)
        .def("SetParameters", &LayerType::SetParameters)
        .def("InitRandomPars", &LayerType::InitRandomPars);
    // TODO add other methods?
  }
  {
    using LayerType = Convolutional<MachineType>;
    py::class_<LayerType, AbLayerType, std::shared_ptr<LayerType>>(
        m, "Convolutional")
        .def(py::init<const AbstractGraph &, AbActivationType &, int, int, int,
                      bool>(),
             py::arg("graph"), py::arg("activation"), py::arg("input_channels"),
             py::arg("output_channels"), py::arg("distance") = 1,
             py::arg("use_bias") = false)
        .def("Ninput", &LayerType::Ninput)
        .def("Noutput", &LayerType::Noutput)
        .def("Npar", &LayerType::Npar)
        .def("GetParameters", &LayerType::GetParameters)
        .def("SetParameters", &LayerType::SetParameters)
        .def("InitRandomPars", &LayerType::InitRandomPars);
    // TODO add other methods?
  }
  {
    using LayerType = SumOutput<MachineType>;
    py::class_<LayerType, AbLayerType, std::shared_ptr<LayerType>>(m,
                                                                   "SumOutput")
        .def(py::init<int>(), py::arg("input_size"))
        .def("Ninput", &LayerType::Ninput)
        .def("Noutput", &LayerType::Noutput)
        .def("Npar", &LayerType::Npar)
        .def("GetParameters", &LayerType::GetParameters)
        .def("SetParameters", &LayerType::SetParameters)
        .def("InitRandomPars", &LayerType::InitRandomPars);
    // TODO add other methods?
  }
  {
    using WfType = FFNN<MachineType>;
    py::class_<WfType, AbMachineType>(m, "FFNN")
        .def(py::init<const AbstractHilbert &,
                      std::vector<std::shared_ptr<AbLayerType>>>(),
             py::arg("hilbert"), py::arg("layers"))
        .def("Npar", &WfType::Npar)
        .def("GetParameters", &WfType::GetParameters)
        .def("SetParameters", &WfType::SetParameters)
        .def("InitRandomPars", &WfType::InitRandomPars, py::arg("seed"),
             py::arg("sigma"))
        .def("Nvisible", &WfType::Nvisible)
        .def("GetHilbert", &WfType::GetHilbert);
    // TODO add other methods?
  }

  // Samplers
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

  py::class_<MetropolisLocalPt<AbMachineType>, SamplerType>(m,
                                                            "MetropolisLocalPt")
      .def(py::init<AbMachineType &, int>(), py::arg("machine"),
           py::arg("nreplicas"))
      .def("Reset", &MetropolisLocalPt<AbMachineType>::Reset)
      .def("Sweep", &MetropolisLocalPt<AbMachineType>::Sweep)
      .def("Visible", &MetropolisLocalPt<AbMachineType>::Visible)
      .def("SetVisible", &MetropolisLocalPt<AbMachineType>::SetVisible)
      .def("Psi", &MetropolisLocalPt<AbMachineType>::Psi)
      .def("Acceptance", &MetropolisLocalPt<AbMachineType>::Acceptance);

  py::class_<MetropolisHop<AbMachineType>, SamplerType>(m, "MetropolisHop")
      .def(py::init<AbstractGraph &, AbMachineType &, int>(), py::arg("graph"),
           py::arg("machine"), py::arg("dmax"))
      .def("Reset", &MetropolisHop<AbMachineType>::Reset)
      .def("Sweep", &MetropolisHop<AbMachineType>::Sweep)
      .def("Visible", &MetropolisHop<AbMachineType>::Visible)
      .def("SetVisible", &MetropolisHop<AbMachineType>::SetVisible)
      .def("Psi", &MetropolisHop<AbMachineType>::Psi)
      .def("Acceptance", &MetropolisHop<AbMachineType>::Acceptance);

  using MetroHamType =
      MetropolisHamiltonian<AbMachineType, AbstractHamiltonian>;
  py::class_<MetroHamType, SamplerType>(m, "MetropolisHamiltonian")
      .def(py::init<AbMachineType &, AbstractHamiltonian &>(),
           py::arg("machine"), py::arg("hamiltonian"))
      .def("Reset", &MetroHamType::Reset)
      .def("Sweep", &MetroHamType::Sweep)
      .def("Visible", &MetroHamType::Visible)
      .def("SetVisible", &MetroHamType::SetVisible)
      .def("Psi", &MetroHamType::Psi)
      .def("Acceptance", &MetroHamType::Acceptance);

  using MetroHamPtType =
      MetropolisHamiltonianPt<AbMachineType, AbstractHamiltonian>;
  py::class_<MetroHamPtType, SamplerType>(m, "MetropolisHamiltonianPt")
      .def(py::init<AbMachineType &, AbstractHamiltonian &, int>(),
           py::arg("machine"), py::arg("hamiltonian"), py::arg("nreplicas"))
      .def("Reset", &MetroHamPtType::Reset)
      .def("Sweep", &MetroHamPtType::Sweep)
      .def("Visible", &MetroHamPtType::Visible)
      .def("SetVisible", &MetroHamPtType::SetVisible)
      .def("Psi", &MetroHamPtType::Psi)
      .def("Acceptance", &MetroHamPtType::Acceptance);

  using MetroExType = MetropolisExchange<AbMachineType>;
  py::class_<MetroExType, SamplerType>(m, "MetropolisExchange")
      .def(py::init<const AbstractGraph &, AbMachineType &, int>(),
           py::arg("graph"), py::arg("machine"), py::arg("dmax") = 1)
      .def("Reset", &MetroExType::Reset)
      .def("Sweep", &MetroExType::Sweep)
      .def("Visible", &MetroExType::Visible)
      .def("SetVisible", &MetroExType::SetVisible)
      .def("Psi", &MetroExType::Psi)
      .def("Acceptance", &MetroExType::Acceptance);

  {
    using DerSampler = MetropolisExchangePt<AbMachineType>;
    py::class_<DerSampler, SamplerType>(m, "MetropolisExchangePt")
        .def(py::init<const AbstractGraph &, AbMachineType &, int, int>(),
             py::arg("graph"), py::arg("machine"), py::arg("dmax") = 1,
             py::arg("nreplicas") = 1)
        .def("Reset", &DerSampler::Reset)
        .def("Sweep", &DerSampler::Sweep)
        .def("Visible", &DerSampler::Visible)
        .def("SetVisible", &DerSampler::SetVisible)
        .def("Psi", &DerSampler::Psi)
        .def("Acceptance", &DerSampler::Acceptance);
  }

  {
    using DerSampler = ExactSampler<AbMachineType>;
    py::class_<DerSampler, SamplerType>(m, "ExactSampler")
        .def(py::init<AbMachineType &>(), py::arg("machine"))
        .def("Reset", &DerSampler::Reset)
        .def("Sweep", &DerSampler::Sweep)
        .def("Visible", &DerSampler::Visible)
        .def("SetVisible", &DerSampler::SetVisible)
        .def("Psi", &DerSampler::Psi)
        .def("Acceptance", &DerSampler::Acceptance);
  }

  {
    using DerSampler = CustomSampler<AbMachineType>;
    using MatType = DerSampler::MatType;
    py::class_<DerSampler, SamplerType>(m, "CustomSampler")
        .def(py::init<AbMachineType &, const std::vector<MatType> &,
                      const std::vector<std::vector<int>> &,
                      std::vector<double>>(),
             py::arg("machine"), py::arg("move_operators"),
             py::arg("acting_on"),
             py::arg("move_weights") = std::vector<double>())
        .def("Reset", &DerSampler::Reset)
        .def("Sweep", &DerSampler::Sweep)
        .def("Visible", &DerSampler::Visible)
        .def("SetVisible", &DerSampler::SetVisible)
        .def("Psi", &DerSampler::Psi)
        .def("Acceptance", &DerSampler::Acceptance);
  }

  {
    using DerSampler = CustomSamplerPt<AbMachineType>;
    using MatType = DerSampler::MatType;
    py::class_<DerSampler, SamplerType>(m, "CustomSamplerPt")
        .def(py::init<AbMachineType &, const std::vector<MatType> &,
                      const std::vector<std::vector<int>> &,
                      std::vector<double>, int>(),
             py::arg("machine"), py::arg("move_operators"),
             py::arg("acting_on"),
             py::arg("move_weights") = std::vector<double>(),
             py::arg("nreplicas"))
        .def("Reset", &DerSampler::Reset)
        .def("Sweep", &DerSampler::Sweep)
        .def("Visible", &DerSampler::Visible)
        .def("SetVisible", &DerSampler::SetVisible)
        .def("Psi", &DerSampler::Psi)
        .def("Acceptance", &DerSampler::Acceptance);
  }

  py::class_<AbstractOptimizer>(m, "Optimizer");

  py::class_<Sgd, AbstractOptimizer>(m, "Sgd").def(
      py::init<double, double, double>(), py::arg("learning_rate"),
      py::arg("l2reg") = 0, py::arg("decay_factor") = 1.0);
  // TODO add other methods?

  py::class_<VariationalMonteCarlo>(m, "Vmc")
      .def(py::init<AbstractHamiltonian &, SamplerType &, AbstractOptimizer &,
                    int, int, std::string, int, int, std::string, double, bool,
                    bool, bool, int>(),
           py::arg("hamiltonian"), py::arg("sampler"), py::arg("optimizer"),
           py::arg("nsamples"), py::arg("niter_opt"), py::arg("output_file"),
           py::arg("discarded_samples") = -1,
           py::arg("discarded_samples_on_init") = 0, py::arg("method") = "Sr",
           py::arg("diag_shift") = 0.01, py::arg("rescale_shift") = false,
           py::arg("use_iterative") = false, py::arg("use_cholesky") = true,
           py::arg("save_every") = 50)

      .def("Run", &VariationalMonteCarlo::Run);

  py::class_<eddetail::result_t>(m, "EdResult")
      .def_readwrite("eigenvalues", &eddetail::result_t::eigenvalues)
      .def_readwrite("eigenvectors", &eddetail::result_t::eigenvectors)
      .def_readwrite("which_eigenvector",
                     &eddetail::result_t::which_eigenvector);

  m.def("LanczosEd", &lanczos_ed, py::arg("hamiltonian"),
        py::arg("matrix_free") = false, py::arg("first_n") = 1,
        py::arg("max_iter") = 1000, py::arg("seed") = 42,
        py::arg("precision") = 1.0e-14, py::arg("get_groundstate") = false);
}  // namespace netket

}  // namespace netket

#endif
