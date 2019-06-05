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

#ifndef NETKET_PYRMSPROP_HPP
#define NETKET_PYRMSPROP_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "rms_prop.hpp"

namespace py = pybind11;

namespace netket {

void AddRmsProp(py::module &subm) {
  py::class_<RMSProp, AbstractOptimizer>(subm, "RmsProp", R"EOF(
    RMSProp is a well-known update algorithm proposed by Geoff Hinton
    in his Neural Networks course notes [Neural Networks course notes](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
    It corrects the problem with AdaGrad by using an exponentially weighted
    moving average over past squared gradients instead of a cumulative sum.
    After initializing the vector $$\mathbf{s}$$ to zero, $$s_k$$ and t
    he parameters $$p_k$$ are updated as

    $$
    \begin{align}
    s^\prime_k = \beta s_k + (1-\beta) G_k(\mathbf{p})^2 \\
    p^\prime_k = p_k - \frac{\eta}{\sqrt{s_k}+\epsilon} G_k(\mathbf{p})
    \end{align}
    $$)EOF")
      .def(py::init<double, double, double>(), py::arg("learning_rate") = 0.001,
           py::arg("beta") = 0.9, py::arg("epscut") = 1.0e-7, R"EOF(
           Constructs a new ``RmsProp`` optimizer.

           Args:
               learning_rate: The learning rate $$ \eta $$
               beta: Exponential decay rate.
               epscut: Small cutoff value.

           Examples:
               RmsProp optimizer.

               ```python
               >>> from netket.optimizer import RmsProp
               >>> op = RmsProp(learning_rate=0.02)

               ```
           )EOF");
}

}  // namespace netket

#endif
