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
}

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
      .def_property_readonly("machine", &AbstractSampler::GetMachine, R"EOF(
        netket.machine: The machine used for the sampling.  )EOF")
      .def_property_readonly("batch_size", &AbstractSampler::BatchSize, R"EOF(
        int: Number of samples in a batch.)EOF")
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


  py::class_<MCResult>(subm, "MCResult",
                       R"EOF(Result of Monte Carlo sampling.)EOF")
      .def_property_readonly(
          "samples",
          [](const MCResult &self) {
            assert(self.samples.rows() % self.n_chains == 0);
            return detail::as_readonly(py::array_t<double, py::array::c_style>{
                {self.samples.rows() / self.n_chains, self.n_chains,
                 self.samples.cols()},
                self.samples.data(),
                py::none()});
          },
          py::return_value_policy::reference_internal,
          R"EOF(Visible configurations `{vᵢ}` visited during sampling.)EOF")
      .def_property_readonly(
          "log_values",
          [](const MCResult &self) {
            assert(self.log_values.rows() % self.n_chains == 0);
            return detail::as_readonly(py::array_t<Complex, py::array::c_style>{
                {self.log_values.rows() / self.n_chains, self.n_chains},
                self.log_values.data(),
                py::none()});
          },
          py::return_value_policy::reference_internal,
          R"EOF(An array of `complex128` representing `Ψ(vᵢ)` for all
                sampled visible configurations `vᵢ`.)EOF");

  subm.def(
      "compute_samples",
      [](AbstractSampler &sampler, Index n_samples, Index n_discard) {
        // Helper types and lambda for writing the shapes in a clean way:
        using Shape2 = std::array<std::size_t, 2>;
        using Shape3 = std::array<std::size_t, 3>;
        const auto _ = [](Index i) { return static_cast<std::size_t>(i); };

        MCResult result =
            ComputeSamples(sampler, n_samples, n_discard,
            /*der_logs=*/nonstd::nullopt);

        const Shape3 sample_shape = {_(result.samples.rows() / result.n_chains),
                                     _(result.n_chains),
                                     _(result.samples.cols())};
        auto samples = py::array_t<double, py::array::c_style>{
            sample_shape, result.samples.data()};

        const Shape2 logval_shape = {_(result.samples.rows() / result.n_chains),
                                     _(result.n_chains)};
        auto logvals = py::array_t<Complex, py::array::c_style>{
            logval_shape, result.log_values.data()};

        return py::make_tuple(samples, logvals);
      },
      py::arg{"sampler"}, py::arg{"n_samples"}, py::arg{"n_discard"},
      R"EOF(Runs Monte Carlo sampling using `sampler`.

                  First `n_discard` sweeps are discarded. Results of the next
                  `≥n_samples` sweeps are saved. Since samplers work with
                  batches of specified size it may be impossible to sample
                  exactly `n_samples` visible configurations (without throwing
                  away useful data, of course). You can rely on
                  `compute_samples` to return at least `n_samples` samples.

                  Exact number of performed sweeps and samples stored can be
                  computed using the following functions:

                  ```python

                  def number_sweeps(sampler, n_samples):
                      return (n_samples + sampler.batch_size - 1) // sampler.batch_size

                  def number_samples(sampler, n_samples):
                      return sampler.batch_size * number_sweeps(sampler, n_samples)
                  ```

                  Args:
                      sampler: sampler to use for Monte Carlo sweeps.
                      n_samples: number of samples to record.
                      n_discard: number of sweeps to discard.
                      der_logs: Whether to calculate gradients of the logarithm
                          of the wave function. `None` means don't compute,
                          "normal" means compute, and "centered" means compute
                          and then center.

                  Returns:
                      A `MCResult` object with all the data obtained during sampling.)EOF");

}

}  // namespace netket

#endif
