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

#ifndef NETKET_PYADAGRAD_HPP
#define NETKET_PYADAGRAD_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "ada_grad.hpp"

namespace py = pybind11;

namespace netket {

void AddAdaGrad(py::module &subm) {
  py::class_<AdaGrad, AbstractOptimizer>(subm, "AdaGrad",
                                         R"EOF(AdaGrad Optimizer.
    In many cases, in Sgd the learning rate $$\eta$$ should
    decay as a function of training iteration to prevent overshooting
    as the optimum is approached. AdaGrad is an adaptive learning
    rate algorithm that automatically scales the learning rate with a sum
    over past gradients. The vector $$\mathbf{g}$$ is initialized to zero.
    Given a stochastic estimate of the gradient of the cost function $$G(\mathbf{p})$$,
    the updates for $$g_k$$ and the parameter $$p_k$$ are

    $$
    \begin{aligned}
    g^\prime_k &= g_k + G_k(\mathbf{p})^2\\
    p^\prime_k &= p_k - \frac{\eta}{\sqrt{g_k + \epsilon}}G_k(\mathbf{p})
    \end{aligned}
    $$

    AdaGrad has been shown to perform particularly well when
    the gradients are sparse, but the learning rate may become too small
    after many updates because the sum over the squares of past gradients is cumulative.

  )EOF")
      .def(py::init<double, double>(), py::arg("learning_rate") = 0.001,
           py::arg("epscut") = 1.0e-7, R"EOF(
           Constructs a new ``AdaGrad`` optimizer.

           Args:
               learning_rate: Learning rate $$\eta$$.
               epscut: Small $$\epsilon$$ cutoff.

           Examples:
               Simple AdaDelta optimizer.

               ```python
               >>> from netket.optimizer import AdaGrad
               >>> op = AdaGrad()

               ```
           )EOF");
}

}  // namespace netket

#endif
