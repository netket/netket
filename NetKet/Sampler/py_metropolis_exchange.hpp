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

#ifndef NETKET_PY_METROPOLISEXCHANGE_HPP
#define NETKET_PY_METROPOLISEXCHANGE_HPP

#include <pybind11/pybind11.h>
#include "metropolis_exchange.hpp"

namespace py = pybind11;

namespace netket {

void AddMetropolisExchange(py::module &subm) {
  py::class_<MetropolisExchange, AbstractSampler>(subm, "MetropolisExchange",
                                                  R"EOF(
    This sampler acts locally only on two local degree of freedom $$ s_i $$ and $$ s_j $$,
    and proposes a new state: $$ s_1 \dots s^\prime_i \dots s^\prime_j \dots s_N $$,
    where in general $$ s^\prime_i \neq s_i $$ and $$ s^\prime_j \neq s_j $$ .
    The sites $$ i $$ and $$ j $$ are also chosen to be within a maximum graph
    distance of $$ d_{\mathrm{max}} $$.

    The transition probability associated to this sampler can
    be decomposed into two steps:

    1. A pair of indices $$ i,j = 1\dots N $$, and such
    that $$ \mathrm{dist}(i,j) \leq d_{\mathrm{max}} $$,
    is chosen with uniform probability.
    2. The sites are exchanged, i.e. $$ s^\prime_i = s_j $$ and $$ s^\prime_j = s_i $$.

    Notice that this sampling method generates random permutations of the quantum
    numbers, thus global quantities such as the sum of the local quantum n
    umbers are conserved during the sampling.
    This scheme should be used then only when sampling in a
    region where $$ \sum_i s_i = \mathrm{constant} $$ is needed,
    otherwise the sampling would be strongly not ergodic.
    )EOF")
      .def(py::init<const AbstractGraph &, AbstractMachine &, int>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("graph"),
           py::arg("machine"), py::arg("d_max") = 1, R"EOF(
             Constructs a new ``MetropolisExchange`` sampler given a machine and a
             graph.

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
                 >>> print(sa.hilbert.size)
                 100

                 ```
             )EOF");
}
}  // namespace netket
#endif
