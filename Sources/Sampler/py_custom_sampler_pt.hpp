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

#ifndef NETKET_PY_CUSTOMSAMPLERPT_HPP
#define NETKET_PY_CUSTOMSAMPLERPT_HPP

#include <pybind11/pybind11.h>
#include "custom_sampler_pt.hpp"

namespace py = pybind11;

namespace netket {

void AddCustomSamplerPt(py::module &subm) {
  py::class_<CustomSamplerPt, AbstractSampler>(subm, "CustomSamplerPt")
      .def(py::init<AbstractMachine &, const LocalOperator &,
                    const std::vector<double> &, int>(),
           py::keep_alive<1, 2>(), py::arg("machine"),
           py::arg("move_operators"),
           py::arg("move_weights") = std::vector<double>(),
           py::arg("n_replicas") = 1, R"EOF(
             Constructs a new ``CustomSamplerPt`` given a machine and a list of local
             stochastic move (transition) operators.

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
                 >>> # Construct a Custom Sampler with parallel tempering
                 >>> # Using random local spin flips (Pauli X operator)
                 >>> X = [[0, 1],[1, 0]]
                 >>> move_op = nk.operator.LocalOperator(hilbert=hi,operators=[X] * g.n_sites,acting_on=[[i] for i in range(g.n_sites)])
                 >>> sa = nk.sampler.CustomSamplerPt(machine=ma, move_operators=move_op,n_replicas=10)

                 ```
             )EOF");
}
}  // namespace netket
#endif
