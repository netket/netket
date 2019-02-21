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

#ifndef NETKET_PYADAMAX_HPP
#define NETKET_PYADAMAX_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "ada_max.hpp"

namespace py = pybind11;

namespace netket {

void AddAdaMax(py::module &subm) {
  py::class_<AdaMax, AbstractOptimizer>(subm, "AdaMax", R"EOF(AdaMax Optimizer.
    AdaMax is an adaptive stochastic gradient descent method,
    and a variant of [Adam](https://arxiv.org/pdf/1412.6980.pdf) based on the infinity norm.
    In contrast to the SGD, AdaMax offers the important advantage of being much
    less sensitive to the choice of the hyper-parameters (for example, the learning rate).

    Given a stochastic estimate of the gradient of the cost function ($$ G(\mathbf{p}) $$),
    AdaMax performs an update:

    $$
    p^\prime_k = p_k + \mathcal{S}_k,
    $$

    where $$ \mathcal{S}_k $$ implicitly depends on all the history of the optimization up to the current point.
    The NetKet naming convention of the parameters strictly follows the one introduced by the authors of AdaMax.
    For an in-depth description of this method, please refer to
    [Kingma, D., & Ba, J. (2015). Adam: a method for stochastic optimization](https://arxiv.org/pdf/1412.6980.pdf)
    (Algorithm 2 therein).)EOF")
      .def(py::init<double, double, double, double>(), py::arg("alpha") = 0.001,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("epscut") = 1.0e-7, R"EOF(
           Constructs a new ``AdaMax`` optimizer.

           Args:
               alpha: The step size.
               beta1: First exponential decay rate.
               beta2: Second exponential decay rate.
               epscut: Small epsilon cutoff.

           Examples:
               Simple AdaMax optimizer.

               ```python
               >>> from netket.optimizer import AdaMax
               >>> op = AdaMax()

               ```
           )EOF");
}

}  // namespace netket

#endif
