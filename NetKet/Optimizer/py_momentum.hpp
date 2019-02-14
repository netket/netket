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

#ifndef NETKET_PYMOMENTUM_HPP
#define NETKET_PYMOMENTUM_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "momentum.hpp"

namespace py = pybind11;

namespace netket {

void AddMomentum(py::module &subm) {
  py::class_<Momentum, AbstractOptimizer>(subm, "Momentum",
                                          R"EOF(Momentum-based Optimizer.
      The momentum update incorporates an exponentially weighted moving average
      over previous gradients to speed up descent
      [Qian, N. (1999)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf).
      The momentum vector $$\mathbf{m}$$ is initialized to zero.
      Given a stochastic estimate of the gradient of the cost function
      $$G(\mathbf{p})$$, the updates for the parameter $$p_k$$ and
      corresponding component of the momentum $$m_k$$ are

      $$
      \begin{aligned}
      m^\prime_k &= \beta m_k + (1-\beta)G_k(\mathbf{p})\\
      p^\prime_k &= \eta m^\prime_k
      \end{aligned}
      $$)EOF")
      .def(py::init<double, double>(), py::arg("learning_rate") = 0.001,
           py::arg("beta") = 0.9, R"EOF(
           Constructs a new ``Momentum`` optimizer.

           Args:
               learning_rate: The learning rate $$ \eta $$
               beta: Momentum exponential decay rate, should be in [0,1].

           Examples:
               Momentum optimizer.

               ```python
               >>> from netket.optimizer import Momentum
               >>> op = Momentum(learning_rate=0.01)

               ```
           )EOF");
}

}  // namespace netket

#endif
