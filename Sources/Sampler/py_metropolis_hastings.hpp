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

#ifndef NETKET_PY_METROPOLIS_HASTINGS_HPP
#define NETKET_PY_METROPOLIS_HASTINGS_HPP

#include <pybind11/pybind11.h>
#include "exchange_kernel.hpp"
#include "metropolis_hastings.hpp"

namespace py = pybind11;

namespace netket {

void AddMetropolisHastings(py::module& subm) {
  py::class_<MetropolisHastings, AbstractSampler>(subm, "MetropolisHastings",
                                                  R"EOF(

                 )EOF")
      .def(py::init([](AbstractMachine& machine,
                       MetropolisHastings::TransitionKernel tk,
                       Index batch_size, nonstd::optional<Index> sweep_size) {
             return MetropolisHastings(machine, tk, batch_size,
                                       sweep_size.value_or(machine.Nvisible()));
           }),
           py::keep_alive<1, 2>(), py::arg("machine"),
           py::arg("transition_kernel"), py::arg("batch_size") = 16,
           py::arg("sweep_size") = py::none(),
           R"EOF(
             Constructs a new ``MetropolisHastings`` sampler given a machine and
             a transition kernel.

             Args:
                 machine: A machine $$\Psi(s)$$ used for the sampling.
                          The probability distribution being sampled
                          from is $$F(\Psi(s))$$, where the function
                          $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
                 graph: A graph used to define the distances among the degrees
                        of freedom being sampled.
                 d_max: The maximum graph distance allowed for exchanges.

             Examples:
                 Sampling from a RBM machine in a 1D lattice of spin 1/2, using
                 nearest-neighbours exchanges.

                 ```python
                 >>> import netket as nk
                 >>>
                 >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
                 >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
                 >>>
                 >>> # RBM Spin Machine
                 >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
                 >>>
                 >>> # Construct a MetropolisExchange Sampler
                 >>> sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=1)
                 >>> print(sa.machine.hilbert.size)
                 100

                 ```
             )EOF");

  py::class_<MetropolisHastingsPt, AbstractSampler>(subm,
                                                    "MetropolisHastingsPt",
                                                    R"EOF(

                 )EOF")
      .def(py::init([](AbstractMachine& machine,
                       MetropolisHastings::TransitionKernel tk,
                       Index n_replicas, nonstd::optional<Index> sweep_size) {
             return MetropolisHastingsPt(
                 machine, tk, n_replicas,
                 sweep_size.value_or(machine.Nvisible()));
           }),
           py::keep_alive<1, 2>(), py::arg("machine"),
           py::arg("transition_kernel"), py::arg("n_replicas") = 16,
           py::arg("sweep_size") = py::none(),
           R"EOF(
                        This sampler performs parallel-tempering
                        moves in addition to the moves implemented in `MetropolisHastings`.
                        The number of replicas can be $$ N_{\mathrm{rep}} $$ chosen by the user.

                        Args:
                        machine: A machine $$\Psi(s)$$ used for the sampling.
                                 The probability distribution being sampled
                                 from is $$F(\Psi(s))$$, where the function
                                 $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
                        n_replicas: The number of replicas used for parallel tempering.
                )EOF");
}
}  // namespace netket
#endif
