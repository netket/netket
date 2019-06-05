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

#ifndef NETKET_PYAMSGRAD_HPP
#define NETKET_PYAMSGRAD_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "ams_grad.hpp"

namespace py = pybind11;

namespace netket {

void AddAmsGrad(py::module &subm) {
  py::class_<AMSGrad, AbstractOptimizer>(subm, "AmsGrad",
                                         R"EOF(AmsGrad Optimizer.
    In some cases, adaptive learning rate methods such as AdaMax fail
    to converge to the optimal solution because of the exponential
    moving average over past gradients. To address this problem,
    Sashank J. Reddi, Satyen Kale and Sanjiv Kumar proposed the
    AmsGrad [update algorithm](https://openreview.net/forum?id=ryQu7f-RZ).
    The update rule for $$\mathbf{v}$$ (equivalent to $$E[g^2]$$ in AdaDelta
    and $$\mathbf{s}$$ in RMSProp) is modified such that $$v^\prime_k \geq v_k$$
    is guaranteed, giving the algorithm a "long-term memory" of past gradients.
    The vectors $$\mathbf{m}$$ and $$\mathbf{v}$$ are initialized to zero, and
    are updated with the parameters $$\mathbf{p}$$:

    $$
    \begin{aligned}
    m^\prime_k &= \beta_1 m_k + (1-\beta_1)G_k(\mathbf{p})\\
    v^\prime_k &= \beta_2 v_k + (1-\beta_2)G_k(\mathbf{p})^2\\
    v^\prime_k &= \mathrm{Max}(v^\prime_k, v_k)\\
    p^\prime_k &= p_k - \frac{\eta}{\sqrt{v^\prime_k}+\epsilon}m^\prime_k
    \end{aligned}
    $$)EOF")
      .def(py::init<double, double, double, double>(),
           py::arg("learning_rate") = 0.001, py::arg("beta1") = 0.9,
           py::arg("beta2") = 0.999, py::arg("epscut") = 1.0e-7, R"EOF(
           Constructs a new ``AmsGrad`` optimizer.

           Args:
               learning_rate: The learning rate $\eta$.
               beta1: First exponential decay rate.
               beta2: Second exponential decay rate.
               epscut: Small epsilon cutoff.

           Examples:
               Simple AmsGrad optimizer.

               ```python
               >>> from netket.optimizer import AmsGrad
               >>> op = AmsGrad()

               ```
           )EOF");
}

}  // namespace netket

#endif
