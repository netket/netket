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
#include "exchange_kernel.hpp"
#include "metropolis_hastings.hpp"

namespace py = pybind11;

namespace netket {

void AddMetropolisExchange(py::module &subm) {
  subm.def(
      "MetropolisExchange",
      [](AbstractMachine &m, nonstd::optional<AbstractGraph *> g, Index dmax,
         Index batch_size, nonstd::optional<Index> sweep_size) {
        if (g.has_value()) {
          WarningMessage()
              << "graph argument is deprecated and does not have any effect "
                 "here. The graph is deduced automatically from machine.\n";
        }
        return MetropolisHastings(m, ExchangeKernel{m, dmax},
                                  sweep_size.value_or(m.Nvisible()),
                                  batch_size);
      },
      py::keep_alive<1, 2>(), py::arg("machine"), py::arg("graph") = py::none(),
      py::arg("d_max") = 1, py::arg("batch_size") = 16,
      py::arg{"sweep_size"} = py::none(),
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


          Args:
              machine: A machine $$\Psi(s)$$ used for the sampling.
                       The probability distribution being sampled
                       from is $$F(\Psi(s))$$, where the function
                       $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.

              graph: DEPRECATED argument
              sweep_size: The number of exchanges that compose a single sweep.
                          If not specified, sweep_size is equal to the number of degrees of freedom (n_visible).
              batch_size: The number of Markov Chain to be run in parallel on a single process.
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
              >>> sa = nk.sampler.MetropolisExchange(machine=ma)
              >>> print(sa.machine.hilbert.size)
              100

              ```


          )EOF");

  // AddAcceptance(cls);
}
}  // namespace netket
#endif
