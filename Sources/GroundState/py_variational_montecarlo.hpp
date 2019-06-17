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

void AddVariationalMonteCarloModule(py::module &m) {
  auto m_vmc = m.def_submodule("variational");

  py::class_<VariationalMonteCarlo>(
      m_vmc, "Vmc",
      R"EOF(Variational Monte Carlo schemes to learn the ground state using stochastic reconfiguration and gradient descent optimizers.)EOF")
      .def(py::init<const AbstractOperator &, AbstractSampler &,
                    AbstractOptimizer &, int, int, int, const std::string &,
                    const std::string &, double, bool, bool>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
           py::keep_alive<1, 4>(), py::arg("hamiltonian"), py::arg("sampler"),
           py::arg("optimizer"), py::arg("n_samples"),
           py::arg("discarded_samples") = -1,
           py::arg("discarded_samples_on_init") = 0,
           py::arg("target") = "energy", py::arg("method") = "Sr",
           py::arg("diag_shift") = 0.01, py::arg("use_iterative") = false,
           py::arg("use_cholesky") = true,
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
               use_cholesky: Whether to use cholesky decomposition. The default
                   is true.

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
      .def_property_readonly("vmc_data", &VariationalMonteCarlo::GetVmcData);

  py::class_<vmc::Result>(m_vmc, "_VmcResult");

  m_vmc.def("compute_samples", vmc::ComputeSamples, py::arg("sampler"),
            py::arg("nsamples"), py::arg("ndiscard") = 0,
            py::arg("compute_logderivs") = true, R"EOF(
           Computes a sequence of visible configurations based on Monte Carlo sampling using `sampler`.

           Args:
               sampler: The sampler used to perform the MC sweeps.
               nsamples: The number of MC samples that are stored (one MC sweep
                    is performed between each sample)
               ndiscard: The number of sweeps to be discarded before starting
                    to store samples.
               compute_logderivs: Whether to store the logarithmic derivatives
                    of the wavefunction as part of the returned VMC result.
            )EOF");

  m_vmc.def(
      "expectation",
      [](const vmc::Result &result, AbstractMachine &psi,
         const AbstractOperator &op, bool return_locvals) {
        if (return_locvals) {
          VectorXcd locvals;
          auto ex = vmc::Expectation(result, psi, op, locvals);
          return static_cast<py::object>(py::make_tuple(ex, locvals));
        } else {
          return py::cast(vmc::Expectation(result, psi, op));
        }
      },
      py::arg("vmc_data"), py::arg("psi"), py::arg("op"),
      py::arg("return_locvals") = false, R"EOF(
           Computes the expectation value of a Hermitian operator based on
           provided VMC data.

           Args:
               vmc_data: The VMC result data.
               psi: Machine represenation of the wavefunction.
               op: Hermitian operator.
               return_locvals: If `True`, this function will additionally
                   return an array containing the local values of the observable
                   for all visible configurations in `vmc_data`.

            Examples:
               A very basic VMC loop in Python:

               ```python
                 from netket.graph import Hypercube
                 from netket.hilbert import Spin
                 from netket.operator import Ising
                 from netket.machine import RbmSpin
                 from netket.sampler import MetropolisLocal
                 from netket.variational import compute_samples, expectation, gradient

                 hi = Spin(s=0.5, graph=Hypercube(8, 1))
                 ham = Ising(hi, h=1.0)
                 psi = RbmSpin(hi, alpha=2)
                 psi.init_random_parameters(sigma=0.1)
                 sampler = MetropolisLocal(psi)

                 for step in range(10):
                     data = compute_samples(sampler, 10000, 1000)

                     ex = expectation(data, psi, ham)
                     print("E={Mean:.4f} ± {Sigma:.4f}".format(**ex))

                     grad = gradient(data, psi, ham)
                     psi.parameters -= 0.1 * grad
               ```
            )EOF");

  using VarType1 = vmc::Stats (*)(const vmc::Result &, AbstractMachine &,
                                  const AbstractOperator &);
  m_vmc.def("variance", (VarType1)&vmc::Variance, py::arg("vmc_data"),
            py::arg("psi"), py::arg("op"), R"EOF(
           Computes the variance value of a Hermitian operator, i.e.,
                σ² = ⟨(O - ⟨O⟩)²⟩
           based on provided VMC data.

           Args:
               vmc_data: The VMC result data.
               psi: Machine represenation of the wavefunction.
               op: Hermitian operator.
            )EOF");

  using VarType2 =
      vmc::Stats (*)(const vmc::Result &, AbstractMachine &,
                     const AbstractOperator &, double, const VectorXcd &);
  m_vmc.def("variance", (VarType2)&vmc::Variance, py::arg("vmc_data"),
            py::arg("op"), py::arg("psi"), py::arg("expectation_value"),
            py::arg("locvals"), R"EOF(
           Computes the variance value of a Hermitian operator, i.e.,
                σ² = ⟨(O - ⟨O⟩)²⟩
           based on provided VMC data, reusing precomputed local values
           and expectation of `op`.

           Args:
               vmc_data: The VMC result data.
               psi: Machine represenation of the wavefunction.
               op: Hermitian operator.
               expectation_value: The expectation of `op` for the given
                   VMC data.
               locvals: An array containing the local values of `op` for the
                   given VMC data.

            Examples:
               Computing the variance using precomputed values.

               ```python
                 from netket.graph import Hypercube
                 from netket.hilbert import Spin
                 from netket.operator import Ising
                 from netket.machine import RbmSpin
                 from netket.sampler import MetropolisLocal
                 import netket.variational as vmc

                 hi = Spin(s=0.5, graph=Hypercube(8, 1))
                 ham = Ising(hi, h=1.0)
                 psi = RbmSpin(hi, alpha=2)
                 psi.init_random_parameters(sigma=0.1)
                 sampler = MetropolisLocal(psi)

                 data = vmc.compute_samples(sampler, 10000, 1000)
                 ex, lv = vmc.expectation(data, psi, ham, return_locvals=True)
                 var = vmc.variance(data, psi, ham, ex["Mean"], lv)
               ```
            )EOF");

  using GradType1 = VectorXcd (*)(const vmc::Result &, AbstractMachine &,
                                  const AbstractOperator &);
  m_vmc.def("gradient", (GradType1)&vmc::Gradient, py::arg("vmc_data"),
            py::arg("op"), py::arg("psi"), R"EOF(
           Computes the gradient of the expecation value of a Hermitian operator
           `op` with respect to the wavefunction parameters based on provided VMC
           data.

           Args:
               vmc_data: The VMC result data.
               psi: Machine represenation of the wavefunction.
               op: Hermitian operator.
            )EOF");

  using GradType2 = VectorXcd (*)(const vmc::Result &, AbstractMachine &,
                                  const AbstractOperator &, const VectorXcd &);
  m_vmc.def("gradient", (GradType2)&vmc::Gradient, py::arg("vmc_data"),
            py::arg("op"), py::arg("psi"), py::arg("locvals"), R"EOF(
           Computes the gradient of the expecation value of a Hermitian operator
           `op` with respect to the wavefunction parameters based on provided VMC
           data.

           Args:
               vmc_data: The VMC result data.
               psi: Machine represenation of the wavefunction.
               op: Hermitian operator.
               locvals: An array containing the local values of `op` for the
                   given VMC data.
            )EOF");

  m_vmc.def("local_value", &vmc::LocalValue, py::arg("op"), py::arg("psi"),
            py::arg("v"), R"EOF(
           Computes the local value of the operator `op` in configuration `v`
           which is defined as O_loc(v) = ⟨v|op|Ψ⟩ / ⟨v|Ψ⟩.

           Args:
               op: Hermitian operator.
               psi: Machine represenation of the wavefunction.
               v: Visible configuration.
            )EOF");

  m_vmc.def(
      "local_values",
      [](const vmc::Result &result, AbstractMachine &psi,
         const AbstractOperator &op) {
        return vmc::LocalValues(op, psi, result.SampleMatrix());
      },
      py::arg("vmc_data"), py::arg("psi"), py::arg("op"), R"EOF(
           Computes the local values of the operator `op` for all visible
           configurations stored in `vmc_data`.

           Args:
               vmc_data: The VMC result data.
               psi: Machine represenation of the wavefunction.
               op: Hermitian operator.
            )EOF");
}

}  // namespace netket

#endif
