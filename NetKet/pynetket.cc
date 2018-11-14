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

#include "Dynamics/pydynamics.hpp"
#include "Graph/pygraph.hpp"
#include "Hilbert/pyhilbert.hpp"
#include "Machine/pymachine.hpp"
#include "Operator/pyoperator.hpp"
#include "Output/pyoutput.hpp"
#include "Stats/binning.hpp"
#include "netket.hpp"

namespace py = pybind11;

// PYBIND11_MAKE_OPAQUE(std::vector<int>);
// PYBIND11_MAKE_OPAQUE(std::vector<double>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::complex<double>>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<int>>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<double>>);

namespace netket {

using ode::AddDynamicsModule;

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

  AddDynamicsModule(m);
  AddGraphModule(m);
  AddHilbertModule(m);
  AddMachineModule(m);
  AddOperatorModule(m);
  AddOutputModule(m);

  // Samplers
  using MachineType = std::complex<double>;
  using AbMachineType = AbstractMachine<MachineType>;
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

  py::class_<ImaginaryTimeDriver>(m, "ImaginaryTimeDriver")
      .def(py::init<ImaginaryTimeDriver::Matrix &,
                    ImaginaryTimeDriver::Stepper &, JsonOutputWriter &, double,
                    double, double>(),
           py::arg("hamiltonian"), py::arg("stepper"), py::arg("output_writer"),
           py::arg("tmin"), py::arg("tmax"), py::arg("dt"))
      .def("add_observable", &ImaginaryTimeDriver::AddObservable,
           py::arg("observable"), py::arg("name"),
           py::arg("matrix_type") = "Sparse")
      .def("run", &ImaginaryTimeDriver::Run, py::arg("initial_state"));

  py::class_<eddetail::result_t>(m, "EdResult")
      .def_readwrite("eigenvalues", &eddetail::result_t::eigenvalues)
      .def_readwrite("eigenvectors", &eddetail::result_t::eigenvectors)
      .def_readwrite("which_eigenvector",
                     &eddetail::result_t::which_eigenvector);

  m.def("LanczosEd", &lanczos_ed, py::arg("operator"),
        py::arg("matrix_free") = false, py::arg("first_n") = 1,
        py::arg("max_iter") = 1000, py::arg("seed") = 42,
        py::arg("precision") = 1.0e-14, py::arg("get_groundstate") = false);

}  // PYBIND11_MODULE

}  // namespace netket

#endif
