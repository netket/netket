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

#ifndef NETKET_PY_CUSTOMSAMPLER_HPP
#define NETKET_PY_CUSTOMSAMPLER_HPP

#include <pybind11/pybind11.h>
#include "custom_local_kernel.hpp"
#include "metropolis_hastings.hpp"

namespace py = pybind11;

namespace netket {

void AddCustomSampler(py::module subm) {
  subm.def(
      "CustomSampler",
      [](AbstractMachine &m, const LocalOperator &move_operators,
         const std::vector<double> &move_weights, Index batch_size,
         nonstd::optional<Index> sweep_size) {
        return MetropolisHastings(
            m, CustomLocalKernel{m, move_operators, move_weights}, batch_size,
            sweep_size.value_or(m.Nvisible()));
      },
      py::keep_alive<0, 1>(), py::arg("machine"), py::arg("move_operators"),
      py::arg("move_weights") = std::vector<double>(),
      py::arg("batch_size") = 16, py::arg{"sweep_size"} = py::none(), R"EOF(
      Custom Sampler, where transition operators are specified by the user.
      For the moment, this functionality is limited to transition operators which
      are sums of $$k$$-local operators:

      $$
      \mathcal{M}= \sum_i M_i
      $$

      where the move operators $$ M_i $$ act on an (arbitrary) subset of sites.

      The operators $$ M_i $$ are specified giving their matrix elements, and a list
      of sites on which they act. Each operator $$ M_i $$ must be real,
      symmetric, positive definite and stochastic (i.e. sum of each column and line is 1).

      The transition probability associated to a custom sampler can be decomposed into two steps:

      1. One of the move operators $$ M_i $$ is chosen with a weight given by the
      user (or uniform probability by default). If the weights are provided,
      they do not need to sum to unity.

      2. Starting from state
      $$ |n \rangle $$, the probability to transition to state
      $$ |m\rangle $$ is given by
      $$ \langle n|  M_i | m \rangle $$.


       Args:
           machine: A machine $$\Psi(s)$$ used for the sampling.
                  The probability distribution being sampled
                  from is $$F(\Psi(s))$$, where the function
                  $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
           move_operators: The stochastic `LocalOperator`
                $$\mathcal{M}= \sum_i M_i$$ used for transitions.
           move_weights: For each $$ i $$, the probability to pick one of
                the move operators (must sum to one).

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
           >>> # Construct a Custom Sampler
           >>> # Using random local spin flips (Pauli X operator)
           >>> X = [[0, 1],[1, 0]]
           >>> move_op = nk.operator.LocalOperator(hilbert=hi,operators=[X] * g.n_sites,acting_on=[[i] for i in range(g.n_sites)])
           >>> sa = nk.sampler.CustomSampler(machine=ma, move_operators=move_op)

           ```
     )EOF");

  subm.def("CustomSamplerPt",
           [](AbstractMachine &m, const LocalOperator &move_operators,
              const std::vector<double> &move_weights, Index n_replicas,
              nonstd::optional<Index> sweep_size) {
             return MetropolisHastingsPt(
                 m, CustomLocalKernel{m, move_operators, move_weights},
                 n_replicas, sweep_size.value_or(m.Nvisible()));
           },
           py::keep_alive<0, 1>(), py::arg("machine"),
           py::arg("move_operators"),
           py::arg("move_weights") = std::vector<double>(),
           py::arg("n_replicas") = 16, py::arg{"sweep_size"} = py::none(),
           R"EOF(
          This sampler performs parallel-tempering
          moves in addition to the local moves implemented in `CustomSampler`.
          The number of replicas can be $$ N_{\mathrm{rep}} $$ chosen by the user.

          Args:
              machine: A machine $$\Psi(s)$$ used for the sampling.
                       The probability distribution being sampled
                       from is $$F(\Psi(s))$$, where the function
                       $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
              move_operators: The stochastic `LocalOperator`
                       $$\mathcal{M}= \sum_i M_i$$ used for transitions.
              move_weights: For each $$ i $$, the probability to pick one of
                       the move operators (must sum to one).
              n_replicas: The number of replicas used for parallel tempering.
              sweep_size: The number of exchanges that compose a single sweep.
                          If None, sweep_size is equal to the number of degrees of freedom (n_visible).


              ```
              )EOF");
}
}  // namespace netket
#endif
