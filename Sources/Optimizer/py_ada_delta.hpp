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

#ifndef NETKET_PYADADELTA_HPP
#define NETKET_PYADADELTA_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "ada_delta.hpp"

namespace py = pybind11;

namespace netket {

void AddAdaDelta(py::module &subm) {
  py::class_<AdaDelta, AbstractOptimizer>(subm, "AdaDelta",
                                          R"EOF(AdaDelta Optimizer.
    Like RMSProp, [AdaDelta](http://arxiv.org/abs/1212.5701) corrects the
    monotonic decay of learning rates associated with AdaGrad,
    while additionally eliminating the need to choose a global
    learning rate $$ \eta $$. The NetKet naming convention of
    the parameters strictly follows the one introduced in the original paper;
    here $$E[g^2]$$ is equivalent to the vector $$\mathbf{s}$$ from RMSProp.
    $$E[g^2]$$ and $$E[\Delta x^2]$$ are initialized as zero vectors.

    $$
    \begin{align}
    E[g^2]^\prime_k &= \rho E[g^2] + (1-\rho)G_k(\mathbf{p})^2\\
    \Delta p_k &= - \frac{\sqrt{E[\Delta x^2]+\epsilon}}{\sqrt{E[g^2]+ \epsilon}}G_k(\mathbf{p})\\
    E[\Delta x^2]^\prime_k &= \rho E[\Delta x^2] + (1-\rho)\Delta p_k^2\\
    p^\prime_k &= p_k + \Delta p_k\\
    \end{align}
    $$

    )EOF")
      .def(py::init<double, double>(), py::arg("rho") = 0.95,
           py::arg("epscut") = 1.0e-7, R"EOF(
           Constructs a new ``AdaDelta`` optimizer.

           Args:
               rho: Exponential decay rate, in [0,1].
               epscut: Small $$\epsilon$$ cutoff.

           Examples:
               Simple AdaDelta optimizer.

               ```python
               >>> from netket.optimizer import AdaDelta
               >>> op = AdaDelta()

               ```
           )EOF");
}

}  // namespace netket

#endif
