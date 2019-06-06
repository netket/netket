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

#ifndef NETKET_PY_METROPOLISEXCHANGEPT_HPP
#define NETKET_PY_METROPOLISEXCHANGEPT_HPP

#include <pybind11/pybind11.h>
#include "metropolis_exchange_pt.hpp"

namespace py = pybind11;

namespace netket {

void AddMetropolisExchangePt(py::module &subm) {
  py::class_<MetropolisExchangePt, AbstractSampler>(
      subm, "MetropolisExchangePt", R"EOF(
    This sampler performs parallel-tempering moves in addition to
    the local exchange moves implemented in `MetropolisExchange`.
    The number of replicas can be $$ N_{\mathrm{rep}} $$ chosen by the user.
    )EOF")
      .def(py::init<const AbstractGraph &, AbstractMachine &, int, int>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("graph"),
           py::arg("machine"), py::arg("d_max") = 1, py::arg("n_replicas") = 1,
           R"EOF(
             Constructs a new ``MetropolisExchangePt`` sampler given a machine, a
             graph, and a number of replicas.

             Args:
                 machine: A machine $$\Psi(s)$$ used for the sampling.
                          The probability distribution being sampled
                          from is $$F(\Psi(s))$$, where the function
                          $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
                 graph: A graph used to define the distances among the degrees
                        of freedom being sampled.
                 d_max: The maximum graph distance allowed for exchanges.
                 n_replicas: The number of replicas used for parallel tempering.

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
                 >>> # Construct a MetropolisExchange Sampler with parallel tempering
                 >>> sa = nk.sampler.MetropolisExchangePt(machine=ma,graph=g,d_max=1,n_replicas=16)

                 ```
             )EOF");
}

}  // namespace netket
#endif
