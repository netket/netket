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

#ifndef NETKET_PY_METROPOLISHOP_HPP
#define NETKET_PY_METROPOLISHOP_HPP

#include <pybind11/pybind11.h>
#include "hop_kernel.hpp"
#include "metropolis_hastings.hpp"

namespace py = pybind11;

namespace netket {

void AddMetropolisHop(py::module &subm) {
  subm.def("MetropolisHop",
           [](AbstractMachine &m, Index dmax, Index n_chains,
              nonstd::optional<Index> sweep_size,
              nonstd::optional<Index> batch_size) {
             return MetropolisHastings(m, HopKernel{m, dmax}, n_chains,
                                       sweep_size.value_or(m.Nvisible()),
                                       batch_size.value_or(n_chains));
           },
           py::keep_alive<1, 2>(), py::arg("machine"), py::arg("d_max") = 1,
           py::arg("n_chains") = 16, py::arg{"sweep_size"} = py::none(),
           py::arg{"batch_size"} = py::none(),
           R"EOF(
          This sampler acts locally only on two local degree of freedom $$ s_i $$ and $$ s_j $$,
          and proposes a new state picking up uniformely from the local degrees of freedom.
          The resultin state is : $$ s_1 \dots s^\prime_i \dots s^\prime_j \dots s_N $$,
          where in general it is not guarantueed that $$ s^\prime_i \neq s_i $$ and $$ s^\prime_j \neq s_j $$ .
          The sites $$ i $$ and $$ j $$ are also chosen to be within a maximum graph
          distance of $$ d_{\mathrm{max}} $$.

          Args:
              machine: A machine $$\Psi(s)$$ used for the sampling.
                       The probability distribution being sampled
                       from is $$F(\Psi(s))$$, where the function
                       $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.

              d_max: The maximum graph distance allowed for exchanges.
              n_chains: The number of Markov Chain to be run in parallel on a single process.
              sweep_size: The number of exchanges that compose a single sweep.
                          If None, sweep_size is equal to the number of degrees of freedom (n_visible).
              batch_size: The batch size to be used when calling log_val on the given Machine.
                          If None, batch_size is equal to the number Markov chains (n_chains).

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
              >>> # Construct a MetropolisHop Sampler
              >>> sa = nk.sampler.MetropolisHop(machine=ma)
              >>> print(sa.machine.hilbert.size)
              100

              ```


          )EOF");
}
}  // namespace netket
#endif
