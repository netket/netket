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
#include "Hilbert/pyhilbert.hpp"
#include "netket.hpp"

namespace py = pybind11;

// PYBIND11_MAKE_OPAQUE(std::vector<int>);
// PYBIND11_MAKE_OPAQUE(std::vector<double>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::complex<double>>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<int>>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<double>>);

namespace netket {

PYBIND11_MODULE(netket, m) {
  // py::bind_vector<std::vector<int>>(m, "VectorInt");
  // py::bind_vector<std::vector<std::vector<int>>>(m, "VectorVectorInt");
  // py::bind_vector<std::vector<double>>(m, "VectorDouble");
  // py::bind_vector<std::vector<std::vector<double>>>(m, "VectorVectorDouble");
  // py::bind_vector<std::vector<std::complex<double>>>(m,
  // "VectorComplexDouble");
  // TODO move modules in separate files closer to their binding classes

  py::class_<netket::default_random_engine>(m, "RandomEngine")
      .def(py::init<netket::default_random_engine::result_type>(),
           py::arg("seed") = netket::default_random_engine::default_seed)
      .def("Seed", (void (netket::default_random_engine::*)(
                       netket::default_random_engine::result_type)) &
                       netket::default_random_engine::seed);

  py::class_<Lookup<double>>(m, "LookupReal").def(py::init<>());
  py::class_<Lookup<std::complex<double>>>(m, "LookupComplex")
      .def(py::init<>());

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

  AddHilbertModule(m);

  py::class_<AbstractOperator, std::shared_ptr<AbstractOperator>>(m, "Operator")
      .def("GetConn", &AbstractOperator::GetConn)
      .def("GetHilbert", &AbstractOperator::GetHilbert);

  py::class_<LocalOperator, AbstractOperator, std::shared_ptr<LocalOperator>>(
      m, "LocalOperator")
      .def(
          py::init<const AbstractHilbert &, std::vector<LocalOperator::MatType>,
                   std::vector<LocalOperator::SiteType>>(),
          py::arg("hilbert"), py::arg("operators"), py::arg("acting_on"))
      .def(py::init<const AbstractHilbert &, LocalOperator::MatType,
                    LocalOperator::SiteType>(),
           py::arg("hilbert"), py::arg("operator"), py::arg("acting_on"))
      .def("GetConn", &LocalOperator::GetConn)
      .def("GetHilbert", &LocalOperator::GetHilbert)
      .def("LocalMatrices", &LocalOperator::LocalMatrices)
      .def(py::self += py::self)
      .def(py::self *= double())
      .def(py::self *= std::complex<double>())
      .def(py::self * py::self);
  // .def(double() * py::self)
  // .def(py::self * double())
  // .def(std::complex<double>() * py::self)
  // .def(py::self * std::complex<double>());

  py::class_<Ising, AbstractOperator, std::shared_ptr<Ising>>(m, "Ising")
      .def(py::init<const AbstractHilbert &, double, double>(),
           py::arg("hilbert"), py::arg("h"), py::arg("J") = 1.0)
      .def("GetConn", &Ising::GetConn)
      .def("GetHilbert", &Ising::GetHilbert);

  py::class_<Heisenberg, AbstractOperator, std::shared_ptr<Heisenberg>>(
      m, "Heisenberg")
      .def(py::init<const AbstractHilbert &>(), py::arg("hilbert"))
      .def("GetConn", &Heisenberg::GetConn)
      .def("GetHilbert", &Heisenberg::GetHilbert);

  py::class_<GraphHamiltonian, AbstractOperator,
             std::shared_ptr<GraphHamiltonian>>(m, "GraphHamiltonian")
      .def(py::init<const AbstractHilbert &, GraphHamiltonian::OVecType,
                    GraphHamiltonian::OVecType, std::vector<int>>(),
           py::arg("hilbert"),
           py::arg("siteops") = GraphHamiltonian::OVecType(),
           py::arg("bondops") = GraphHamiltonian::OVecType(),
           py::arg("bondops_colors") = std::vector<int>())
      .def("GetConn", &GraphHamiltonian::GetConn)
      .def("GetHilbert", &GraphHamiltonian::GetHilbert);

  py::class_<BoseHubbard, AbstractOperator, std::shared_ptr<BoseHubbard>>(
      m, "BoseHubbard")
      .def(py::init<const AbstractHilbert &, double, double, double>(),
           py::arg("hilbert"), py::arg("U"), py::arg("V") = 0.,
           py::arg("mu") = 0.)
      .def("GetConn", &BoseHubbard::GetConn)
      .def("GetHilbert", &BoseHubbard::GetHilbert);

  py::class_<SparseMatrixWrapper<AbstractOperator>>(m, "SparseMatrixWrapper")
      .def(py::init<const AbstractOperator &>(), py::arg("operator"))
      .def("GetMatrix", &SparseMatrixWrapper<AbstractOperator>::GetMatrix);

  using MachineType = std::complex<double>;
  using AbMachineType = AbstractMachine<MachineType>;
  py::class_<AbMachineType>(m, "Machine")
      .def("Npar", &AbMachineType::Npar)
      .def("GetParameters", &AbMachineType::GetParameters)
      .def("SetParameters", &AbMachineType::SetParameters)
      .def("InitRandomPars", &AbMachineType::InitRandomPars)
      .def("LogVal",
           (MachineType(AbMachineType::*)(AbMachineType::VisibleConstType)) &
               AbMachineType::LogVal)
      .def("LogValDiff", (AbMachineType::VectorType(AbMachineType::*)(
                             AbMachineType::VisibleConstType,
                             const std::vector<std::vector<int>> &,
                             const std::vector<std::vector<double>> &)) &
                             AbMachineType::LogValDiff)
      .def("DerLog", &AbMachineType::DerLog)
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
        .def("LogVal",
             (MachineType(WfType::*)(AbMachineType::VisibleConstType)) &
                 WfType::LogVal)
        .def("LogValDiff", (AbMachineType::VectorType(WfType::*)(
                               AbMachineType::VisibleConstType,
                               const std::vector<std::vector<int>> &,
                               const std::vector<std::vector<double>> &)) &
                               WfType::LogValDiff)
        .def("DerLog", &WfType::DerLog)
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
        .def("LogVal",
             (MachineType(WfType::*)(AbMachineType::VisibleConstType)) &
                 WfType::LogVal)
        .def("LogValDiff", (AbMachineType::VectorType(WfType::*)(
                               AbMachineType::VisibleConstType,
                               const std::vector<std::vector<int>> &,
                               const std::vector<std::vector<double>> &)) &
                               WfType::LogValDiff)
        .def("DerLog", &WfType::DerLog)
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
        .def("LogVal",
             (MachineType(WfType::*)(AbMachineType::VisibleConstType)) &
                 WfType::LogVal)
        .def("LogValDiff", (AbMachineType::VectorType(WfType::*)(
                               AbMachineType::VisibleConstType,
                               const std::vector<std::vector<int>> &,
                               const std::vector<std::vector<double>> &)) &
                               WfType::LogValDiff)
        .def("DerLog", &WfType::DerLog)
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
        .def("LogVal",
             (MachineType(WfType::*)(AbMachineType::VisibleConstType)) &
                 WfType::LogVal)
        .def("LogValDiff", (AbMachineType::VectorType(WfType::*)(
                               AbMachineType::VisibleConstType,
                               const std::vector<std::vector<int>> &,
                               const std::vector<std::vector<double>> &)) &
                               WfType::LogValDiff)
        .def("DerLog", &WfType::DerLog)
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
        .def("LogVal",
             (MachineType(WfType::*)(AbMachineType::VisibleConstType)) &
                 WfType::LogVal)
        .def("LogValDiff", (AbMachineType::VectorType(WfType::*)(
                               AbMachineType::VisibleConstType,
                               const std::vector<std::vector<int>> &,
                               const std::vector<std::vector<double>> &)) &
                               WfType::LogValDiff)
        .def("DerLog", &WfType::DerLog)
        .def("Nvisible", &WfType::Nvisible)
        .def("GetHilbert", &WfType::GetHilbert);
    // TODO add other methods?
  }
  {
    using WfType = MPSPeriodic<MachineType, true>;
    py::class_<WfType, AbMachineType>(m, "MPSPeriodicDiagonal")
        .def(py::init<const AbstractHilbert &, double, int>(),
             py::arg("hilbert"), py::arg("bond_dim"), py::arg("symperiod") = -1)
        .def("Npar", &WfType::Npar)
        .def("GetParameters", &WfType::GetParameters)
        .def("SetParameters", &WfType::SetParameters)
        .def("InitRandomPars", &WfType::InitRandomPars, py::arg("seed"),
             py::arg("sigma"))
        .def("LogVal",
             (MachineType(WfType::*)(AbMachineType::VisibleConstType)) &
                 WfType::LogVal)
        .def("LogValDiff", (AbMachineType::VectorType(WfType::*)(
                               AbMachineType::VisibleConstType,
                               const std::vector<std::vector<int>> &,
                               const std::vector<std::vector<double>> &)) &
                               WfType::LogValDiff)
        .def("DerLog", &WfType::DerLog)
        .def("Nvisible", &WfType::Nvisible)
        .def("GetHilbert", &WfType::GetHilbert);
    // TODO add other methods?
  }
  {
    using WfType = MPSPeriodic<MachineType, false>;
    py::class_<WfType, AbMachineType>(m, "MPSPeriodic")
        .def(py::init<const AbstractHilbert &, double, int>(),
             py::arg("hilbert"), py::arg("bond_dim"), py::arg("symperiod") = -1)
        .def("Npar", &WfType::Npar)
        .def("GetParameters", &WfType::GetParameters)
        .def("SetParameters", &WfType::SetParameters)
        .def("InitRandomPars", &WfType::InitRandomPars, py::arg("seed"),
             py::arg("sigma"))
        .def("LogVal",
             (MachineType(WfType::*)(AbMachineType::VisibleConstType)) &
                 WfType::LogVal)
        .def("LogValDiff", (AbMachineType::VectorType(WfType::*)(
                               AbMachineType::VisibleConstType,
                               const std::vector<std::vector<int>> &,
                               const std::vector<std::vector<double>> &)) &
                               WfType::LogValDiff)
        .def("DerLog", &WfType::DerLog)
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
                      std::vector<std::shared_ptr<AbLayerType>> &>(),
             py::arg("hilbert"), py::arg("layers"))
        .def("Npar", &WfType::Npar)
        .def("GetParameters", &WfType::GetParameters)
        .def("SetParameters", &WfType::SetParameters)
        .def("InitRandomPars", &WfType::InitRandomPars, py::arg("seed"),
             py::arg("sigma"))
        .def("LogVal",
             (MachineType(WfType::*)(AbMachineType::VisibleConstType)) &
                 WfType::LogVal)
        .def("LogValDiff", (AbMachineType::VectorType(WfType::*)(
                               AbMachineType::VisibleConstType,
                               const std::vector<std::vector<int>> &,
                               const std::vector<std::vector<double>> &)) &
                               WfType::LogValDiff)
        .def("DerLog", (AbMachineType::VectorType(WfType::*)(
                           AbMachineType::VisibleConstType)) &
                           WfType::DerLog)
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

  using MetroHamType = MetropolisHamiltonian<AbMachineType, AbstractOperator>;
  py::class_<MetroHamType, SamplerType>(m, "MetropolisHamiltonian")
      .def(py::init<AbMachineType &, AbstractOperator &>(), py::arg("machine"),
           py::arg("hamiltonian"))
      .def("Reset", &MetroHamType::Reset)
      .def("Sweep", &MetroHamType::Sweep)
      .def("Visible", &MetroHamType::Visible)
      .def("SetVisible", &MetroHamType::SetVisible)
      .def("Psi", &MetroHamType::Psi)
      .def("Acceptance", &MetroHamType::Acceptance);

  using MetroHamPtType =
      MetropolisHamiltonianPt<AbMachineType, AbstractOperator>;
  py::class_<MetroHamPtType, SamplerType>(m, "MetropolisHamiltonianPt")
      .def(py::init<AbMachineType &, AbstractOperator &, int>(),
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

  {
    using OptType = RMSProp;
    py::class_<OptType, AbstractOptimizer>(m, "RMSProp")
        .def(py::init<double, double, double>(),
             py::arg("learning_rate") = 0.001, py::arg("beta") = 0.9,
             py::arg("epscut") = 1.0e-7);
    // TODO add other methods?
  }
  {
    using OptType = Momentum;
    py::class_<OptType, AbstractOptimizer>(m, "Momentum")
        .def(py::init<double, double>(), py::arg("learning_rate") = 0.001,
             py::arg("beta") = 0.9);
    // TODO add other methods?
  }
  {
    using OptType = AMSGrad;
    py::class_<OptType, AbstractOptimizer>(m, "AMSGrad")
        .def(py::init<double, double, double, double>(),
             py::arg("learning_rate") = 0.001, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("epscut") = 1.0e-7);
    // TODO add other methods?
  }
  {
    using OptType = AdaMax;
    py::class_<OptType, AbstractOptimizer>(m, "AdaMax")
        .def(py::init<double, double, double, double>(),
             py::arg("alpha") = 0.001, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("epscut") = 1.0e-7);
    // TODO add other methods?
  }
  {
    using OptType = AdaGrad;
    py::class_<OptType, AbstractOptimizer>(m, "AdaGrad")
        .def(py::init<double, double>(), py::arg("learning_rate") = 0.001,
             py::arg("epscut") = 1.0e-7);
    // TODO add other methods?
  }
  {
    using OptType = AdaDelta;
    py::class_<OptType, AbstractOptimizer>(m, "AdaDelta")
        .def(py::init<double, double>(), py::arg("rho") = 0.95,
             py::arg("epscut") = 1.0e-7);
    // TODO add other methods?
  }

  py::class_<VariationalMonteCarlo>(m, "Vmc")
      .def(py::init<AbstractOperator &, SamplerType &, AbstractOptimizer &, int,
                    int, std::string, int, int, std::string, double, bool, bool,
                    bool, int>(),
           py::arg("hamiltonian"), py::arg("sampler"), py::arg("optimizer"),
           py::arg("nsamples"), py::arg("niter_opt"), py::arg("output_file"),
           py::arg("discarded_samples") = -1,
           py::arg("discarded_samples_on_init") = 0, py::arg("method") = "Sr",
           py::arg("diag_shift") = 0.01, py::arg("rescale_shift") = false,
           py::arg("use_iterative") = false, py::arg("use_cholesky") = true,
           py::arg("save_every") = 50)
      .def("AddObservable", &VariationalMonteCarlo::AddObservable)
      .def("Run", &VariationalMonteCarlo::Run);

  py::class_<eddetail::result_t>(m, "EdResult")
      .def_readwrite("eigenvalues", &eddetail::result_t::eigenvalues)
      .def_readwrite("eigenvectors", &eddetail::result_t::eigenvectors)
      .def_readwrite("which_eigenvector",
                     &eddetail::result_t::which_eigenvector);

  m.def("LanczosEd", &lanczos_ed, py::arg("operator"),
        py::arg("matrix_free") = false, py::arg("first_n") = 1,
        py::arg("max_iter") = 1000, py::arg("seed") = 42,
        py::arg("precision") = 1.0e-14, py::arg("get_groundstate") = false);
}  // namespace netket

}  // namespace netket

#endif
