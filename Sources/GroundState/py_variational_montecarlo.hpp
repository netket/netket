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

#ifndef NETKET_PYVARIATIONALMONTECARLO_HPP
#define NETKET_PYVARIATIONALMONTECARLO_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "variational_montecarlo.hpp"

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

void AddVariationalMonteCarloModule(py::module &m) {
  auto m_vmc = m.def_submodule("variational");

  py::class_<MCResult>(m_vmc, "MCResult",
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
                sampled visible configurations `vᵢ`.)EOF")
      .def_property_readonly(
          "der_logs",
          [](const MCResult &self) -> py::object {
            if (self.der_logs.has_value()) {
              assert(self.der_logs->rows() % self.n_chains == 0);
              return detail::as_readonly(
                  py::array_t<Complex, py::array::c_style>{
                      {self.der_logs->rows() / self.n_chains, self.n_chains,
                       self.der_logs->cols()},
                      self.der_logs->data(),
                      py::none()});
            }
            return py::none();
          },
          py::return_value_policy::reference_internal,
          R"EOF(A matrix of logarithmic derivatives.

                Each row in the matrix corresponds to the gradient of
                `Ψ(vᵢ)` with respect to variational parameters.)EOF");

  py::class_<VariationalMonteCarlo>(
      m_vmc, "Vmc",
      R"EOF(Variational Monte Carlo schemes to learn the ground state using
            stochastic reconfiguration and gradient descent optimizers.)EOF")
      .def(py::init<const AbstractOperator &, AbstractSampler &,
                    AbstractOptimizer &, int, int, int, const std::string &,
                    const std::string &, double, bool, nonstd::optional<bool>,
                    const std::string &>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
           py::keep_alive<1, 4>(), py::arg("hamiltonian"), py::arg("sampler"),
           py::arg("optimizer"), py::arg("n_samples"),
           py::arg("discarded_samples") = -1,
           py::arg("discarded_samples_on_init") = 0,
           py::arg("target") = "energy", py::arg("method") = "Sr",
           py::arg("diag_shift") = 0.01, py::arg("use_iterative") = false,
           py::arg("use_cholesky") = nonstd::nullopt,
           py::arg("sr_lsq_solver") = "LLT",
           R"EOF(
           Constructs a ``VariationalMonteCarlo`` object given a hamiltonian,
           sampler, optimizer, and the number of samples.

           Args:
               hamiltonian: The hamiltonian of the system.
               sampler: The sampler object to generate local exchanges.
               optimizer: The optimizer object that determines how the VMC
                   wavefunction is optimized.
               n_samples: Number of Markov Chain Monte Carlo sweeps to be
                   performed at each step of the optimization.
               discarded_samples: Number of sweeps to be discarded at the
                   beginning of the sampling, at each step of the optimization.
                   Default is -1.
               discarded_samples_on_init: Number of sweeps to be discarded in
                   the first step of optimization, at the beginning of the
                   sampling. The default is 0.
               target: The chosen target to minimize.
                   Possible choices are `energy`, and `variance`.
               method: The chosen method to learn the parameters of the
                   wave-function. Possible choices are `Gd` (Regular Gradient descent),
                   and `Sr` (Stochastic reconfiguration a.k.a. natural gradient).
               diag_shift: The regularization parameter in stochastic
                   reconfiguration. The default is 0.01.
               use_iterative: Whether to use the iterative solver in the Sr
                   method (this is extremely useful when the number of
                   parameters to optimize is very large). The default is false.
               use_cholesky: (Deprecated) Use "LLT" solver (see below). If set to False,
                   "ColPivHouseholder" is used. Please use sr_lsq_solver directly in
                   new code.
               sr_lsq_solver: The solver used to solve the least-squares equation
                   in the SR update. Only used if `method == "SR" and not use_iterative`.
                   Available options are "BDCSVD", "ColPivHouseholder", "LDLT", and "LLT".
                   See the [Eigen documentation](https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html)
                   for a description of the available solvers.
                   The default is "LLT".

           Example:
               Optimizing a 1D wavefunction with Variational Mante Carlo.

               ```python
               >>> import netket as nk
               >>> SEED = 3141592
               >>> g = nk.graph.Hypercube(length=8, n_dim=1)
               >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
               >>> ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
               >>> ma.init_random_parameters(seed=SEED, sigma=0.01)
               >>> ha = nk.operator.Ising(hi, h=1.0)
               >>> sa = nk.sampler.MetropolisLocal(machine=ma)
               >>> sa.seed(SEED)
               >>> op = nk.optimizer.Sgd(learning_rate=0.1)
               >>> vmc = nk.variational.Vmc(hamiltonian=ha, sampler=sa,
               ... optimizer=op, n_samples=500)
               >>> print(vmc.machine.n_visible)
               8

               ```

           )EOF")
      .def_property_readonly(
          "machine", &VariationalMonteCarlo::GetMachine,
          R"EOF(netket.machine.Machine: The machine used to express the wavefunction.)EOF")
      .def("add_observable", &VariationalMonteCarlo::AddObservable,
           py::keep_alive<1, 2>(), py::arg("ob"), py::arg("ob_name"), R"EOF(
           Add an observable quantity, that will be calculated at each
           iteration.

           Args:
               ob: The operator form of the observable.
               ob_name: The name of the observable.

           )EOF")
      .def("run", &VariationalMonteCarlo::Run, py::arg("output_prefix"),
           py::arg("n_iter") = nonstd::nullopt, py::arg("step_size") = 1,
           py::arg("save_params_every") = 50, R"EOF(
           Optimize the Vmc wavefunction.

           Args:
               output_prefix: The output file name, without extension.
               n_iter: The maximum number of iterations.
               step_size: Number of iterations performed at a time. Default is 1.
               save_params_every: Frequency to dump wavefunction parameters. The
                   default is 50.

           Examples:
               Running a simple Vmc calculation.


               ```python
               >>> import netket as nk
               >>> SEED = 3141592
               >>> g = nk.graph.Hypercube(length=8, n_dim=1)
               >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
               >>> ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
               >>> ma.init_random_parameters(seed=SEED, sigma=0.01)
               >>> ha = nk.operator.Ising(hi, h=1.0)
               >>> sa = nk.sampler.MetropolisLocal(machine=ma)
               >>> sa.seed(SEED)
               >>> op = nk.optimizer.Sgd(learning_rate=0.1)
               >>> vmc = nk.variational.Vmc(hamiltonian=ha, sampler=sa,
               ... optimizer=op, n_samples=500)
               >>> vmc.run(output_prefix='test', n_iter=1)


               ```

           )EOF")
      .def("reset", &VariationalMonteCarlo::Reset)
      .def("advance", &VariationalMonteCarlo::Advance, py::arg("steps") = 1,
           R"EOF(
           Perform one or several iteration steps of the VMC calculation. In each step,
           energy and gradient will be estimated via VMC and subsequently, the variational
           parameters will be updated according to the configured method.

           Args:
               steps: Number of VMC steps to perform.

           )EOF")
      .def(
          "get_observable_stats",
          [](VariationalMonteCarlo &self) {
            self.ComputeObservables();
            return self.GetObservableStats();
          },
          R"EOF(
        Calculate and return the value of the operators stored as observables.

        )EOF")
      .def_property_readonly("vmc_data", &VariationalMonteCarlo::GetVmcData)
      .def_property_readonly("sr", &VariationalMonteCarlo::GetSR);

  m_vmc.def("compute_samples", &ComputeSamples, py::arg{"sampler"},
            py::arg{"n_samples"}, py::arg{"n_discard"},
            py::arg{"der_logs"} = py::none(),
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

  m_vmc.def(
      "covariance_sv",
      [](py::array_t<Complex, py::array::c_style> s_values,
         py::array_t<Complex, py::array::c_style> v_values) {
        Eigen::Map<const VectorXcd> s_vector{s_values.data(), s_values.size()};

        // Compute S -> S - E[S]
        Complex mean = s_vector.mean();
        MeanOnNodes(mean);

        switch (s_values.ndim()) {
          case 2:
            NETKET_CHECK(v_values.ndim() == 3, InvalidInputError,
                         "v_values has wrong dimension: " << v_values.ndim()
                                                          << "; expected 3.");
            return Gradient(
                s_vector.array() - mean,
                Eigen::Map<const RowMatrix<Complex>>{
                    v_values.data(), v_values.shape(0) * v_values.shape(1),
                    v_values.shape(2)});
          case 1:
            NETKET_CHECK(v_values.ndim() == 2, InvalidInputError,
                         "v_values has wrong dimension: " << v_values.ndim()
                                                          << "; expected 2.");
            return Gradient(
                s_vector.array() - mean,
                Eigen::Map<const RowMatrix<Complex>>{
                    v_values.data(), v_values.shape(0), v_values.shape(1)});
          default:
            NETKET_CHECK(false, InvalidInputError,
                         "s_values has wrong dimension: "
                             << s_values.ndim() << "; expected either 1 or 2.");
        }  // end switch
      },
      py::arg{"s"}.noconvert(), py::arg{"v"}.noconvert(),
      R"EOF(Computes the covariance of two random variables S, V from MCMC
            data. Note that while S has to be a complex scalar variable,
            V = (v[1], ..., v[m]) can be vector-valued. This is because this
            function is designed to be used for computing the gradient of
            expectation values <O>, where S is the local estimator of O and
            V the vector of log-derivatives of the wavefunction.

            This function estimates R = (r[1], ..., r[m]) where
                r[i] = Cov[v[i], S] = 𝔼[conj(v[i]) * (S - 𝔼[S])].

            Args:
                s_values: A vector (or a matrix) of samples of S with shape
                    `(N, M)` where `N` is the number of samples in every Markov
                    Chain and `M` is the number of Markov Chains.
                    A vector is considered to be an `(N, 1)` matrix.
                v_values: A matrix (or a rank-3 array) of samples of V.
                    Each row of the matrix must correspond to one component v[i].
                    For a rank-3 array, its shape is `(N, M, m)` where `N`
                    is the number of samples, `M` is the number of Markov Chains,
                    and `m` is the dimension of V. A `(N, m)` matrix
                    is treated an an `(N, 1, m)` array.

        )EOF");
}

}  // namespace netket

#endif
