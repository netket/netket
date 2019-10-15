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

#ifndef NETKET_PY_METROPOLISHAMILTONIAN_HPP
#define NETKET_PY_METROPOLISHAMILTONIAN_HPP

#include <pybind11/pybind11.h>
#include "hamiltonian_kernel.hpp"
#include "metropolis_hastings.hpp"
#include "metropolis_hastings_pt.hpp"

namespace py = pybind11;

namespace netket {

void AddMetropolisHamiltonian(py::module &subm) {
  subm.def("MetropolisHamiltonian",
           [](AbstractMachine &m, AbstractOperator &ham, Index n_chains,
              nonstd::optional<Index> sweep_size,
              nonstd::optional<Index> batch_size) {
             return MetropolisHastings(m, HamiltonianKernel{m, ham}, n_chains,
                                       sweep_size.value_or(m.Nvisible()),
                                       batch_size.value_or(n_chains));
           },
           py::keep_alive<0, 1>(), py::arg("machine"), py::arg("hamiltonian"),
           py::arg("n_chains") = 16, py::arg{"sweep_size"} = py::none(),
           py::arg{"batch_size"} = py::none(),
           R"EOF(
            Sampling based on the off-diagonal elements of a Hamiltonian (or a generic Operator).
            In this case, the transition matrix is taken to be:

            $$
            T( \mathbf{s} \rightarrow \mathbf{s}^\prime) = \frac{1}{\mathcal{N}(\mathbf{s})}\theta(|H_{\mathbf{s},\mathbf{s}^\prime}|),
            $$

            where $$ \theta(x) $$ is the Heaviside step function, and $$ \mathcal{N}(\mathbf{s}) $$
            is a state-dependent normalization.
            The effect of this transition probability is then to connect (with uniform probability)
            a given state $$ \mathbf{s} $$ to all those states $$ \mathbf{s}^\prime $$ for which the Hamiltonian has
            finite matrix elements.
            Notice that this sampler preserves by construction all the symmetries
            of the Hamiltonian. This is in generally not true for the local samplers instead.



           Args:
               machine: A machine $$\Psi(s)$$ used for the sampling.
                        The probability distribution being sampled
                        from is $$F(\Psi(s))$$, where the function
                        $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
               hamiltonian: The operator used to perform off-diagonal transition.
               n_chains: The number of Markov Chain to be run in parallel on a single process.
               sweep_size: The number of exchanges that compose a single sweep.
                           If None, sweep_size is equal to the number of degrees of freedom (n_visible).
               batch_size: The batch size to be used when calling log_val on the given Machine.
                           If None, batch_size is equal to the number Markov chains (n_chains).                                      

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
               >>> # Transverse-field Ising Hamiltonian
               >>> ha = nk.operator.Ising(hilbert=hi, h=1.0)
               >>>
               >>> # Construct a MetropolisHamiltonian Sampler
               >>> sa = nk.sampler.MetropolisHamiltonian(machine=ma,hamiltonian=ha)

              ```
          )EOF");

  subm.def("MetropolisHamiltonianPt",
           [](AbstractMachine &m, AbstractOperator &ham, Index n_replicas,
              nonstd::optional<Index> sweep_size) {
             return MetropolisHastingsPt(m, HamiltonianKernel{m, ham},
                                         n_replicas,
                                         sweep_size.value_or(m.Nvisible()));
           },
           py::keep_alive<0, 1>(), py::arg("machine"), py::arg("hamiltonian"),
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
                hamiltonian: The operator used to perform off-diagonal transition.
                n_replicas: The number of replicas used for parallel tempering.
                sweep_size: The number of exchanges that compose a single sweep.
                             If None, sweep_size is equal to the number of degrees of freedom (n_visible).
            )EOF");
}
}  // namespace netket
#endif
