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
  auto mh =
      py::class_<MetropolisHastings, AbstractSampler>(subm,
                                                      "MetropolisHastings",
                                                      R"EOF(

            ``MetropolisHastings`` is a generic Metropolis-Hastings sampler using
            a local transition kernel to perform moves in the Markov Chain.
            The transition kernel is used to generate
            a proposed state $$ s^\prime $$, starting from the current state $$ s $$.
            The move is accepted with probability

            $$
            A(s\rightarrow s^\prime) = \mathrm{min}\left (1,\frac{P(s^\prime)}{P(s)} F(e^{L(s,s^\prime)})\right),
            $$

            where the probability being sampled is $$ F(\Psi(s)) $$ (by default $$ F(x)=|x|^2 $$)
            and $L(s,s^\prime)$ is a correcting factor computed by the transition kernel.

                 )EOF")
          .def(py::init([](AbstractMachine& machine,
                           MetropolisHastings::TransitionKernel tk,
                           Index n_chains, nonstd::optional<Index> sweep_size) {
                 return MetropolisHastings(
                     machine, tk, n_chains,
                     sweep_size.value_or(machine.Nvisible()));
               }),
               py::keep_alive<1, 2>(), py::arg("machine"),
               py::arg("transition_kernel"), py::arg("n_chains") = 16,
               py::arg("sweep_size") = py::none(),
               R"EOF(
             Constructs a new ``MetropolisHastings`` sampler given a machine and
             a transition kernel.

             Args:
                 machine: A machine $$\Psi(s)$$ used for the sampling.
                          The probability distribution being sampled
                          from is $$F(\Psi(s))$$, where the function
                          $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
                 transition_kernel: A function to generate a transition.
                          This should take as an input the current state (in batches)
                          and return a modified state (also in batches).
                          This function must also return an array containing the
                          `log_prob_corrections` $$L(s,s^\prime)$$.
                 n_chains: The number of Markov Chain to be run in parallel on a single process.
                 sweep_size: The number of exchanges that compose a single sweep.
                             If None, sweep_size is equal to the number of degrees of freedom (n_visible).

             Examples:
                 Sampling from a RBM machine in a 1D lattice of spin 1/2, using
                 nearest-neighbours exchanges with a custom kernel.

                 ```python
                 import netket as nk
                 import numpy as np

                 # 1D Lattice
                 g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

                 # Hilbert space of spins on the graph
                 # with total Sz equal to 0
                 hi = nk.hilbert.Spin(s=0.5, graph=g, total_sz=0)

                 # Heisenberg hamiltonian
                 ha = nk.operator.Heisenberg(hilbert=hi)

                 # Symmetric RBM Spin Machine
                 ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
                 ma.init_random_parameters(seed=1234, sigma=0.01)

                 # Defining a custom kernel for MetropolisHastings
                 # Notice that this sampler exchanges two random sites
                 # thus preserving the total magnetization
                 # Also notice that it is not recommended to define custom kernels in python
                 # For speed reasons it is better to define exchange kernels using CustomSampler
                 def exchange_kernel(v, vnew, logprobcorr):

                     vnew[:, :] = v[:, :]
                     logprobcorr[:] = 0.0

                     rands = np.random.randint(v.shape[1], size=(v.shape[0], 2))

                     for i in range(v.shape[0]):
                         iss = rands[i, 0]
                         jss = rands[i, 1]

                         vnew[i, iss], vnew[i, jss] = vnew[i, jss], vnew[i, iss]


                 sa = nk.sampler.MetropolisHastings(ma, exchange_kernel, n_chains=16, sweep_size=20)


                 ```
             )EOF");

  AddAcceptance(mh);

  auto mh_pt =
      py::class_<MetropolisHastingsPt, AbstractSampler>(subm,
                                                        "MetropolisHastingsPt",
                                                        R"EOF(
            This sampler performs parallel-tempering
            moves in addition to the moves implemented in `MetropolisHastings`.
            The number of replicas can be $$ N_{\mathrm{rep}} $$ chosen by the user.
            )EOF")
          .def(py::init([](AbstractMachine& machine,
                           MetropolisHastings::TransitionKernel tk,
                           Index n_replicas,
                           nonstd::optional<Index> sweep_size) {
                 return MetropolisHastingsPt(
                     machine, tk, n_replicas,
                     sweep_size.value_or(machine.Nvisible()));
               }),
               py::keep_alive<1, 2>(), py::arg("machine"),
               py::arg("transition_kernel"), py::arg("n_replicas") = 16,
               py::arg("sweep_size") = py::none(),
               R"EOF(
            Constructs a new ``MetropolisHastingsPt`` sampler.

            Args:
                machine: A machine $$\Psi(s)$$ used for the sampling.
                         The probability distribution being sampled
                         from is $$F(\Psi(s))$$, where the function
                         $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
                transition_kernel: A function to generate a transition.
                         This should take as an input the current state (in batches)
                         and return a modified state (also in batches).
                         This function must also return an array containing the
                         `log_prob_corrections` $$L(s,s^\prime)$$.
                n_replicas: The number of replicas used for parallel tempering.
                sweep_size: The number of exchanges that compose a single sweep.
                            If None, sweep_size is equal to the number of degrees
                            of freedom (n_visible).

            )EOF");

  AddAcceptance(mh_pt);
  AddSamplerStats(mh_pt);
}
}  // namespace netket
#endif
