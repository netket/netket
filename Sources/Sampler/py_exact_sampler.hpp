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

#ifndef NETKET_PY_EXACTSAMPLER_HPP
#define NETKET_PY_EXACTSAMPLER_HPP

#include <pybind11/pybind11.h>
#include "exact_sampler.hpp"

namespace py = pybind11;

namespace netket {

void AddExactSampler(py::module &subm) {
  py::class_<ExactSampler, AbstractSampler>(subm, "ExactSampler", R"EOF(
    This sampler generates i.i.d. samples from $$|\Psi(s)|^2$$.
    In order to perform exact sampling, $$|\Psi(s)|^2$$ is precomputed an all
    the possible values of the quantum numbers $$s$$. This sampler has thus an
    exponential cost with the number of degrees of freedom, and cannot be used
    for large systems, where Metropolis-based sampling are instead a viable
    option.
    )EOF")
      .def(py::init<AbstractMachine &>(), py::keep_alive<1, 2>(),
           py::arg("machine"), R"EOF(
             Constructs a new ``ExactSampler`` given a machine.

             Args:
                 machine: A machine $$\Psi(s)$$ used for the sampling.
                          The probability distribution being sampled
                          from is $$F(\Psi(s))$$, where the function
                          $$F(X)$$, is arbitrary, by default $$F(X)=|X|^2$$.

             Examples:
                 Exact sampling from a RBM machine in a 1D lattice of spin 1/2

                 ```python
                 >>> import netket as nk
                 >>>
                 >>> g=nk.graph.Hypercube(length=8,n_dim=1,pbc=True)
                 >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
                 >>>
                 >>> # RBM Spin Machine
                 >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
                 >>>
                 >>> sa = nk.sampler.ExactSampler(machine=ma)

                 ```
             )EOF");
}
}  // namespace netket
#endif
