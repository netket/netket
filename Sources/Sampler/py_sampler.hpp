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

#ifndef NETKET_PY_SAMPLER_HPP
#define NETKET_PY_SAMPLER_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
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
#include "py_custom_sampler.hpp"
#include "py_custom_sampler_pt.hpp"
#include "py_exact_sampler.hpp"
#include "py_metropolis_exchange.hpp"
#include "py_metropolis_exchange_pt.hpp"
#include "py_metropolis_hamiltonian.hpp"
#include "py_metropolis_hamiltonian_pt.hpp"
#include "py_metropolis_hop.hpp"
#include "py_metropolis_local.hpp"
#include "py_metropolis_local_pt.hpp"

namespace py = pybind11;

namespace netket {

void AddSamplerModule(py::module &m) {
  auto subm = m.def_submodule("sampler");

  py::class_<AbstractSampler>(subm, "Sampler", R"EOF(
    NetKet implements generic sampling routines to be used in conjunction with
    suitable variational states, the `Machines`.
    A `Sampler` generates quantum numbers distributed according to:

    $$P(s_1\dots s_N) = F(\Psi(s_1\dots s_N)),$$

    where F is an arbitrary function. By default F(X)=|X|^2.

    The samplers typically transit from the current set of quantum numbers
    $$\mathbf{s} = s_1 \dots s_N$$ to another set
    $$\mathbf{s^\prime} = s^\prime_1 \dots s^\prime_N$$.
    Samplers are then fully specified by the transition probability:

    $$T( \mathbf{s} \rightarrow \mathbf{s}^\prime) .$$
    )EOF")
      .def("seed", &AbstractSampler::Seed, py::arg("base_seed"), R"EOF(
      Seeds the random number generator used by the ``Sampler``.

      Args:
          base_seed: The base seed for the random number generator
          used by the sampler. Each MPI node is guarantueed to be initialized
          with a distinct seed.
      )EOF")
      .def("reset", &AbstractSampler::Reset, py::arg("init_random") = false,
           R"EOF(
      Resets the state of the sampler, including the acceptance rate statistics
      and optionally initializing at random the visible units being sampled.

      Args:
          init_random: If ``True`` the quantum numbers (visible units)
          are initialized at random, otherwise their value is preserved.
      )EOF")
      .def("sweep", &AbstractSampler::Sweep, R"EOF(
      Performs a sampling sweep. Typically a single sweep
      consists of an extensive number of local moves.
      )EOF")
      .def_property("visible", &AbstractSampler::Visible,
                    &AbstractSampler::SetVisible,
                    R"EOF(
                      numpy.array: The quantum numbers being sampled,
                       and distributed according to $$F(\Psi(v))$$ )EOF")
      .def_property_readonly("acceptance", &AbstractSampler::Acceptance, R"EOF(
        numpy.array: The measured acceptance rate for the sampling.
        In the case of rejection-free sampling this is always equal to 1.  )EOF")
      .def_property_readonly("hilbert", &AbstractSampler::GetHilbertShared,
                             R"EOF(
        netket.hilbert: The Hilbert space used for the sampling.  )EOF")
      .def_property_readonly("machine", &AbstractSampler::GetMachine, R"EOF(
        netket.machine: The machine used for the sampling.  )EOF")
      .def_property("machine_func", &AbstractSampler::GetMachineFunc,
                    &AbstractSampler::SetMachineFunc,
                    R"EOF(
                          function(complex): The function to be used for sampling.
                                       by default $$|\Psi(x)|^2$$ is sampled,
                                       however in general $$F(\Psi(v))$$  )EOF");

  AddMetropolisLocal(subm);
  AddMetropolisLocalPt(subm);
  AddMetropolisHop(subm);
  AddMetropolisHamiltonian(subm);
  AddMetropolisHamiltonianPt(subm);
  AddMetropolisExchange(subm);
  AddMetropolisExchangePt(subm);
  AddExactSampler(subm);
  AddCustomSampler(subm);
  AddCustomSamplerPt(subm);
}

}  // namespace netket

#endif
