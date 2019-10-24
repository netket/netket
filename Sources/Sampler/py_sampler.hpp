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

#ifndef NETKET_PY_SAMPLER_HPP
#define NETKET_PY_SAMPLER_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "Graph/graph.hpp"
#include "Operator/operator.hpp"
#include "Sampler/abstract_sampler.hpp"
#include "Utils/memory_utils.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/pybind_helpers.hpp"

namespace netket {
template <class T, class... Args>
pybind11::class_<T, Args...> AddAcceptance(pybind11::class_<T, Args...> cls) {
  return cls.def_property_readonly(
      "acceptance", [](const T& self) { return self.Acceptance(); }, R"EOF(
        float or numpy.array: The measured acceptance rate for the sampling.
        In the case of rejection-free sampling this is always equal to 1.)EOF");
}
template <class T, class... Args>
pybind11::class_<T, Args...> AddSamplerStats(pybind11::class_<T, Args...> cls) {
  return cls.def_property_readonly(
      "stats", [](const T& self) { return self.Stats(); }, R"EOF(
      Internal statistics for the sampling procedure.)EOF");
}
}  // namespace netket

#include "py_custom_sampler.hpp"
#include "py_exact_sampler.hpp"
#include "py_metropolis_exchange.hpp"
#include "py_metropolis_hamiltonian.hpp"
#include "py_metropolis_hastings.hpp"
#include "py_metropolis_hop.hpp"
#include "py_metropolis_local.hpp"

namespace py = pybind11;

namespace netket {

namespace detail {
template <class T, int ExtraFlags>
py::array_t<T, ExtraFlags> as_readonly(py::array_t<T, ExtraFlags> array) {
  py::detail::array_proxy(array.ptr())->flags &=
      ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
  return array;
}
}  // namespace detail

void AddSamplerModule(py::module& m) {
  auto subm = m.def_submodule("sampler");

  py::class_<AbstractSampler>(subm, "Sampler", R"EOF(
    NetKet implements generic sampling routines to be used in conjunction with
    suitable variational states, the `Machines`.
    A `Sampler` generates quantum numbers distributed according to:

    $$P(s_1\dots s_N) = F(\Psi(s_1\dots s_N)),$$

    where F is an arbitrary function. By default F(X)=|X|^2.

    The samplers typically transit from the current set of quantum numbers
    $$\mathbf{s} = s_1 \dots s_N$$ to another set
    $$\mathbf{s^\prime} = s^\prime_1 \dots s^\prime_N$$.
    Samplers are then fully specified by the transition probability:

    $$T( \mathbf{s} \rightarrow \mathbf{s}^\prime) .$$
    )EOF")
      .def("seed", &AbstractSampler::Seed, py::arg("base_seed"), R"EOF(
      Seeds the random number generator used by the ``Sampler``.

      Args:
          base_seed: The base seed for the random number generator
          used by the sampler. Each MPI node is guarantueed to be initialized
          with a distinct seed.
      )EOF")
      .def("reset", &AbstractSampler::Reset, py::arg("init_random") = false,
           R"EOF(
      Resets the state of the sampler, including the acceptance rate statistics
      and optionally initializing at random the visible units being sampled.

      Args:
          init_random: If ``True`` the quantum numbers (visible units)
          are initialized at random, otherwise their value is preserved.
      )EOF")
      .def("sweep", &AbstractSampler::Sweep, R"EOF(
      Performs a sampling sweep. Typically a single sweep
      consists of an extensive number of local moves.
      )EOF")
      .def_property_readonly(
          "visible",
          [](const AbstractSampler& self) { return self.CurrentState().first; },
          R"EOF(A matrix of current visible configurations. Every row
                corresponds to a visible configuration)EOF")
      .def_property_readonly(
          "current_sample",
          [](const AbstractSampler& self) { return self.CurrentState().first; },
          R"EOF(A matrix of current visible configurations. Every row
                          corresponds to a visible configuration)EOF")
      .def_property_readonly(
          "current_state",
          [](const AbstractSampler& self) { return self.CurrentState(); },
          R"EOF(The current sampling state of the sampler. This contains a pair
            (visible,log_val) where log_val is the result of machine.log_val(visible))EOF")
      .def_property_readonly("machine", &AbstractSampler::GetMachine, R"EOF(
        netket.machine: The machine used for the sampling.  )EOF")
      .def_property_readonly("n_chains", &AbstractSampler::BatchSize, R"EOF(
        int: Number of independent chains being sampled.)EOF")
      .def_property(
          "machine_func",
          [](const AbstractSampler& self) {
            return py::cpp_function(
                [&self](py::array_t<Complex, py::array::c_style> x,
                        py::array_t<double, py::array::c_style> out) {
                  const auto input = [&x]() {
                    auto reference = x.unchecked<1>();
                    return nonstd::span<const Complex>{reference.data(0),
                                                       reference.size()};
                  }();
                  const auto output = [&out]() {
                    auto reference = out.mutable_unchecked<1>();
                    return nonstd::span<double>{reference.mutable_data(0),
                                                reference.size()};
                  }();
                  self.GetMachineFunc()(input, output);
                },
                py::arg{"x"}.noconvert(), py::arg{"out"}.noconvert());
          },
          [](AbstractSampler& self, py::function func) {
            self.SetMachineFunc([func](nonstd::span<const Complex> x,
                                       nonstd::span<double> out) {
              auto input = py::array_t<Complex>{static_cast<size_t>(x.size()),
                                                x.data(), /*base=*/py::none()};
              py::detail::array_proxy(input.ptr())->flags &=
                  ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
              auto output = py::array_t<double>{static_cast<size_t>(out.size()),
                                                out.data(),
                                                /*base=*/py::none()};
              func(input, output);
            });
          },
          R"EOF(function(complex): The function to be used for sampling.
                                   by default $$|\Psi(x)|^2$$ is sampled,
                                   however in general $$F(\Psi(v))$$)EOF");

  AddMetropolisLocal(subm);
  AddMetropolisHop(subm);
  AddMetropolisHamiltonian(subm);
  AddMetropolisExchange(subm);
  AddExactSampler(subm);
  AddCustomSampler(subm);
  AddMetropolisHastings(subm);
}

}  // namespace netket

#endif
