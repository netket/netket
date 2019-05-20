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

#ifndef NETKET_PY_METROPOLISLOCALPT_HPP
#define NETKET_PY_METROPOLISLOCALPT_HPP

#include <pybind11/pybind11.h>
#include "metropolis_local_pt.hpp"

namespace py = pybind11;

namespace netket {

void AddMetropolisLocalPt(py::module &subm) {
  py::class_<MetropolisLocalPt, AbstractSampler>(subm, "MetropolisLocalPt",
                                                 R"EOF(
    This sampler performs parallel-tempering
    moves in addition to the local moves implemented in `MetropolisLocal`.
    The number of replicas can be $$ N_{\mathrm{rep}} $$ chosen by the user.
      )EOF")
      .def(py::init<AbstractMachine &, int>(), py::keep_alive<1, 2>(),
           py::arg("machine"), py::arg("n_replicas") = 1, R"EOF(
             Constructs a new ``MetropolisLocalPt`` sampler given a machine
             and the number of replicas.

             Args:
                 machine: A machine $$\Psi(s)$$ used for the sampling.
                          The probability distribution being sampled
                          from is $$F(\Psi(s))$$, where the function
                          $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
                 n_replicas: The number of replicas used for parallel tempering.

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
                 >>> print(sa.hilbert.size)
                 100

                 ```
             )EOF");
}
}  // namespace netket
#endif
