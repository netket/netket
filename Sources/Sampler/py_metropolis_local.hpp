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

#ifndef NETKET_PY_METROPOLISLOCAL_HPP
#define NETKET_PY_METROPOLISLOCAL_HPP

#include <pybind11/pybind11.h>
#include "local_kernel.hpp"
#include "metropolis_hastings.hpp"
#include "metropolis_hastings_pt.hpp"

namespace py = pybind11;

namespace netket {

void AddMetropolisLocal(py::module &subm) {
  subm.def("MetropolisLocal",
           [](AbstractMachine &m, Index batch_size,
              nonstd::optional<Index> sweep_size) {
             return MetropolisHastings(m, LocalKernel{m}, batch_size,
                                       sweep_size.value_or(m.Nvisible()));
           },
           py::keep_alive<0, 1>(), py::arg("machine"),
           py::arg("batch_size") = 16, py::arg{"sweep_size"} = py::none(),
           R"EOF(

             Constructs a new ``MetropolisLocal`` sampler given a machine.

             This sampler acts locally only on one local degree of freedom $$s_i$$,
             and proposes a new state: $$ s_1 \dots s^\prime_i \dots s_N $$,
             where $$ s^\prime_i \neq s_i $$.

             The transition probability associated to this
             sampler can be decomposed into two steps:

             1. One of the site indices $$ i = 1\dots N $$ is chosen
             with uniform probability.
             2. Among all the possible ($$m$$) values that $$s_i$$ can take,
             one of them is chosen with uniform probability.

             For example, in the case of spin $$1/2$$ particles, $$m=2$$
             and the possible local values are $$s_i = -1,+1$$.
             In this case then `MetropolisLocal` is equivalent to flipping a random spin.

             In the case of bosons, with occupation numbers
             $$s_i = 0, 1, \dots n_{\mathrm{max}}$$, `MetropolisLocal`
             would pick a random local occupation number uniformly between $$0$$
             and $$n_{\mathrm{max}}$$.


             Args:
                machine: A machine $$\Psi(s)$$ used for the sampling.
                          The probability distribution being sampled
                          from is $$F(\Psi(s))$$, where the function
                          $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
                batch_size: The number of Markov Chain to be run in parallel on a single process.
                sweep_size: The number of exchanges that compose a single sweep.
                            If None, sweep_size is equal to the number of degrees of freedom (n_visible).


             Examples:
                 Sampling from a RBM machine in a 1D lattice of spin 1/2

                 ```python
                 >>> import netket as nk
                 >>>
                 >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
                 >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
                 >>>
                 >>> # RBM Spin Machine
                 >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
                 >>>
                 >>> # Construct a MetropolisLocal Sampler
                 >>> sa = nk.sampler.MetropolisLocal(machine=ma)
                 >>> print(sa.machine.hilbert.size)
                 100

                 ```
             )EOF");

  subm.def("MetropolisLocalPt",
           [](AbstractMachine &m, Index n_replicas,
              nonstd::optional<Index> sweep_size) {
             return MetropolisHastingsPt(m, LocalKernel{m}, n_replicas,
                                         sweep_size.value_or(m.Nvisible()));
           },
           py::keep_alive<0, 1>(), py::arg("machine"),
           py::arg("n_replicas") = 16, py::arg{"sweep_size"} = py::none(),
           R"EOF(
             This sampler performs parallel-tempering
             moves in addition to the local moves implemented in `MetropolisLocal`.
             The number of replicas can be $$ N_{\mathrm{rep}} $$ chosen by the user.

             Args:
                 machine: A machine $$\Psi(s)$$ used for the sampling.
                          The probability distribution being sampled
                          from is $$F(\Psi(s))$$, where the function
                          $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
                 n_replicas: The number of replicas used for parallel tempering.
                 sweep_size: The number of exchanges that compose a single sweep.
                             If None, sweep_size is equal to the number of degrees of freedom (n_visible).

             Examples:
                 Sampling from a RBM machine in a 1D lattice of spin 1/2

                 ```python
                 >>> import netket as nk
                 >>>
                 >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
                 >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
                 >>>
                 >>> # RBM Spin Machine
                 >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
                 >>>
                 >>> # Construct a MetropolisLocalPt Sampler
                 >>> sa = nk.sampler.MetropolisLocalPt(machine=ma,n_replicas=16)
                 >>> print(sa.machine.hilbert.size)
                 100

                 ```
           )EOF");
  // AddAcceptance(cls);
}
}  // namespace netket
#endif
