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
#include "metropolis_hamiltonian.hpp"

namespace py = pybind11;

namespace netket {

void AddMetropolisHamiltonian(py::module &subm) {
  using DerSampler = MetropolisHamiltonian<AbstractOperator>;
  py::class_<DerSampler, AbstractSampler>(subm, "MetropolisHamiltonian", R"EOF(
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
    )EOF")
      .def(py::init<AbstractMachine &, AbstractOperator &>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("machine"),
           py::arg("hamiltonian"), R"EOF(
             Constructs a new ``MetropolisHamiltonian`` sampler given a machine
             and a Hamiltonian operator (or in general an arbitrary Operator).

             Args:
                 machine: A machine $$\Psi(s)$$ used for the sampling.
                          The probability distribution being sampled
                          from is $$F(\Psi(s))$$, where the function
                          $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
                 hamiltonian: The operator used to perform off-diagonal transition.

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
}
}  // namespace netket
#endif
