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

#ifndef NETKET_PY_METROPOLISHAMILTONIANPT_HPP
#define NETKET_PY_METROPOLISHAMILTONIANPT_HPP

#include <pybind11/pybind11.h>
#include "metropolis_hamiltonian_pt.hpp"

namespace py = pybind11;

namespace netket {

void AddMetropolisHamiltonianPt(py::module &subm) {
  using DerSampler = MetropolisHamiltonianPt<AbstractOperator>;
  py::class_<DerSampler, AbstractSampler>(subm, "MetropolisHamiltonianPt",
                                          R"EOF(
    This sampler performs parallel-tempering moves in addition to
    the local moves implemented in `MetropolisHamiltonian`.
    The number of replicas can be $$ N_{\mathrm{rep}} $$ chosen by the user.
    )EOF")
      .def(py::init<AbstractMachine &, AbstractOperator &, int>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("machine"),
           py::arg("hamiltonian"), py::arg("n_replicas"), R"EOF(
             Constructs a new ``MetropolisHamiltonianPt`` sampler given a machine,
             a Hamiltonian operator (or in general an arbitrary Operator), and the
             number of replicas.

             Args:
                 machine: A machine $$\Psi(s)$$ used for the sampling.
                        The probability distribution being sampled
                        from is $$F(\Psi(s))$$, where the function
                        $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.
                 hamiltonian: The operator used to perform off-diagonal transition.
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
                 >>> # Transverse-field Ising Hamiltonian
                 >>> ha = nk.operator.Ising(hilbert=hi, h=1.0)
                 >>>
                 >>> # Construct a MetropolisHamiltonianPt Sampler
                 >>> sa = nk.sampler.MetropolisHamiltonianPt(machine=ma,hamiltonian=ha,n_replicas=10)

                 ```
             )EOF");
}
}  // namespace netket
#endif
