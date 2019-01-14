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

#ifndef NETKET_PYGROUND_STATE_HPP
#define NETKET_PYGROUND_STATE_HPP

#include "Utils/exceptions.hpp"
#include "ground_state.hpp"
#include <complex>
#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

namespace netket {

void AddGroundStateModule(py::module &m) {
  auto m_exact = m.def_submodule("exact");
  auto m_vmc = m.def_submodule("variational");

  py::class_<VariationalMonteCarlo>(
      m_vmc, "Vmc",
      R"EOF(Variational Monte Carlo schemes to learn the ground state using stochastic reconfiguration and gradient descent optimizers.)EOF")
      .def(py::init<const AbstractOperator &, SamplerType &,
                    AbstractOptimizer &, int, int, int, const std::string &,
                    double, bool, bool, bool>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
           py::keep_alive<1, 4>(), py::arg("hamiltonian"), py::arg("sampler"),
           py::arg("optimizer"), py::arg("n_samples"),
           py::arg("discarded_samples") = -1,
           py::arg("discarded_samples_on_init") = 0, py::arg("method") = "Sr",
           py::arg("diag_shift") = 0.01, py::arg("rescale_shift") = false,
           py::arg("use_iterative") = false, py::arg("use_cholesky") = true,
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
               method: The chosen method to learn the parameters of the 
                   wave-function. The default is `Sr` (stochastic 
                   reconfiguration).
               diag_shift: The regularization parameter in stochastic 
                   reconfiguration. The default is 0.01.
               rescale_shift: Whether to rescale the variational parameters. The 
                   default is false.
               use_iterative: Whether to use the iterative solver in the Sr 
                   method (this is extremely useful when the number of 
                   parameters to optimize is very large). The default is false.
               use_cholesky: Whether to use cholesky decomposition. The default
                   is true.

           Example:
               Optimizing a 1D wavefunction with Variational Mante Carlo.

               ```python
               >>> import netket as nk
               >>> from mpi4py import MPI
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
           py::keep_alive<1, 2>(), R"EOF(
           Add an observable quantity, that will be calculated at each 
           iteration.

           Args:
               ob: The operator form of the observable.
               obname: The name of the observable.

           )EOF")
      .def("run", &VariationalMonteCarlo::Run, py::arg("output_prefix"),
           py::arg("n_iter") = nonstd::nullopt, py::arg("step_size") = 1,
           py::arg("save_params_every") = 50, R"EOF(
           Optimize the Vmc wavefunction.

           Args:
               output_prefix: The output file name, without extension.
               n_iter: The maximum number of iterations.
               step_size: Number of iterations performed at a time. Default is 
                   1.
               save_params_every: Frequency to dump wavefunction parameters. The
                   default is 50.

           Examples:
               Running a simple Vmc calculation.

               
               ```python
               >>> import netket as nk
               >>> from mpi4py import MPI
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
      .def("iter", &VariationalMonteCarlo::Iterate,
           py::arg("n_iter") = nonstd::nullopt, py::arg("step_size") = 1, R"EOF(
           Iterate the optimization of the Vmc wavefunction. 
 
           Args:
               n_iter: The maximum number of iterations.
               step_size: Number of iterations performed at a time. Default is 
                   1.

           )EOF")
      .def("get_observable_stats",
           [](VariationalMonteCarlo &self) {
             py::dict data;
             self.ComputeObservables();
             self.GetObsManager().InsertAllStats(data);
             return data;
           },
           R"EOF(
        Calculate and return the value of the operators stored as observables.

        )EOF");

  py::class_<VariationalMonteCarlo::Iterator>(m_vmc, "VmcIterator")
      .def("__iter__", [](VariationalMonteCarlo::Iterator &self) {
        return py::make_iterator(self.begin(), self.end());
      });

  py::class_<ImagTimePropagation>(
      m_exact, "ImagTimePropagation",
      R"EOF(Solving for the ground state of the wavefunction using imaginary time propagation.)EOF")
      .def(py::init<ImagTimePropagation::Matrix &,
                    ImagTimePropagation::Stepper &, double,
                    ImagTimePropagation::StateVector>(),
           py::arg("hamiltonian"), py::arg("stepper"), py::arg("t0"),
           py::arg("initial_state"), R"EOF(
           Constructs an ``ImagTimePropagation`` object from a hamiltonian, a stepper, 
           a time, and an initial state.

           Args:
               hamiltonian: The hamiltonian of the system. 
               stepper: Stepper (i.e. propagator) that transforms the state of 
                   the system from one timestep to the next.
               t0: The initial time.
               initial_state: The initial state of the system (when propagation 
                   begins.)

           Examples:
               Solving 1D ising model with imagniary time propagation.

               ```python
               >>> from mpi4py import MPI
               >>> import netket as nk
               >>> import numpy as np
               >>> L = 20
               >>> graph = nk.graph.Hypercube(L, n_dim=1, pbc=True)
               >>> hilbert = nk.hilbert.Spin(graph, 0.5)
               >>> hamiltonian = nk.operator.Ising(hilbert, h=1.0)
               >>> mat = nk.operator.wrap_as_matrix(hamiltonian)
               >>> stepper = nk.dynamics.create_timestepper(mat.dimension, rel_tol=1e-10, abs_tol=1e-10)
               >>> output = nk.output.JsonOutputWriter('test.log', 'test.wf')
               >>> psi0 = np.random.rand(mat.dimension)
               >>> driver = nk.exact.ImagTimePropagation(mat, stepper, t0=0, initial_state=psi0)
               >>> driver.add_observable(hamiltonian, 'Hamiltonian')
               >>> for step in driver.iter(dt=0.05, n_iter=2):
               ...     obs = driver.get_observable_stats()

               ```
           )EOF")
      .def("add_observable", &ImagTimePropagation::AddObservable,
           py::keep_alive<1, 2>(), py::arg("observable"), py::arg("name"),
           py::arg("matrix_type") = "Sparse", R"EOF(
           Add an observable quantity, that will be calculated at each 
           iteration.

           Args:
               ob: The operator form of the observable.
               obname: The name of the observable.
               matrix_type: The type of matrix used for the observable when
                   creating the matrix wrapper. The default is `Sparse`. The
                   other choices are `Dense` and `Direct`.

           )EOF")
      .def("iter", &ImagTimePropagation::Iterate, py::arg("dt"),
           py::arg("n_iter") = nonstd::nullopt, R"EOF(
           Iterate the optimization of the Vmc wavefunction. 
 
           Args:
               dt: Number of iterations performed at a time.
               n_iter: The maximum number of iterations.

           )EOF")
      .def_property("t", &ImagTimePropagation::GetTime,
                    &ImagTimePropagation::SetTime,
                    R"EOF(double: Time in the simulation.)EOF")
      .def("get_observable_stats",
           [](const ImagTimePropagation &self) {
             py::dict data;
             self.GetObsManager().InsertAllStats(data);
             return data;
           },
           R"EOF(
        Calculate and return the value of the operators stored as observables.

        )EOF");

  py::class_<ImagTimePropagation::Iterator>(m_exact, "ImagTimeIterator")
      .def("__iter__", [](ImagTimePropagation::Iterator &self) {
        return py::make_iterator(self.begin(), self.end());
      });

  py::class_<eddetail::result_t>(
      m_exact, "EdResult",
      R"EOF(Exact diagonalization of the system hamiltonian using either Lanczos or full diagonalization.)EOF")
      .def_property_readonly(
          "eigenvalues", &eddetail::result_t::eigenvalues,
          R"EOF(vector<double>: The eigenvalues of the hamiltonian of the system.)EOF")
      .def_property_readonly(
          "eigenvectors", &eddetail::result_t::eigenvectors,
          R"EOF(vector<Eigen::Matrix<Complex, Eigen::Dynamic, 1>>: The complex eigenvectors of the system hamiltonian.)EOF")
      .def("mean",
           [](eddetail::result_t &self, AbstractOperator &op, int which) {
             if (which < 0 || static_cast<std::size_t>(which) >=
                                  self.eigenvectors().size()) {
               throw InvalidInputError("Invalid eigenvector index `which`");
             }
             return self.mean(op, which);
           },
           py::arg("operator"), py::arg("which") = 0);

  m_exact.def("lanczos_ed", &lanczos_ed, py::arg("operator"),
              py::arg("matrix_free") = false, py::arg("first_n") = 1,
              py::arg("max_iter") = 1000, py::arg("seed") = 42,
              py::arg("precision") = 1.0e-14,
              py::arg("compute_eigenvectors") = false, R"EOF(
              Use the Lanczos algorithm to diagonalize the operator using 
              routines from IETL.

              Args:
                  operator: The operator to diagnolize.
                  matrix_free: Indicate whether the operator is stored 
                      (sparse/dense) or not (direct). The default is `False`.
                  first_n: The numver of eigenvalues to converge. The default is
                      1.
                  max_iter: The maximum number of iterations. The default is 1000.
                  seed: The random number generator seed. The default is 42.
                  precision: The precision to which the eigenvalues will be 
                      computed. The default is 1e-14.
                  comput_eigenvectors: Whether or not to compute the 
                      eigenvectors of the operator.  The default is `False`.


              Examples:
                  Testing the numer of eigenvalues saved when solving a simple 
                  1D Ising problem.

                  ```python
                  >>> import netket as nk
                  >>> from mpi4py import MPI
                  >>> first_n=3
                  >>> g = nk.graph.Hypercube(length=8, n_dim=1, pbc=True)
                  >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
                  >>> ha = nk.operator.Ising(h=1.0, hilbert=hi)
                  >>> res = nk.exact.lanczos_ed(ha, first_n=first_n, compute_eigenvectors=True)
                  >>> print(len(res.eigenvalues) == first_n)
                  True

                  ```

              )EOF");

  m_exact.def("full_ed", &full_ed, py::arg("operator"), py::arg("first_n") = 1,
              py::arg("compute_eigenvectors") = false, R"EOF(
              Diagonalize the operator using routines from IETL.

              Args:
                  operator: The operator to diagnolize.
                  first_n: The numver of eigenvalues to converge. The default is
                      1.
                  comput_eigenvectors: Whether or not to compute the 
                      eigenvectors of the operator.  The default is `False`.


              Examples:
                  Testing the numer of eigenvalues saved when solving a simple 
                  1D Ising problem.

                  ```python
                  >>> import netket as nk
                  >>> from mpi4py import MPI
                  >>> first_n=3
                  >>> g = nk.graph.Hypercube(length=8, n_dim=1, pbc=True)
                  >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
                  >>> ha = nk.operator.Ising(h=1.0, hilbert=hi)
                  >>> res = nk.exact.full_ed(ha, first_n=first_n, compute_eigenvectors=True)
                  >>> print(len(res.eigenvalues) == first_n)
                  True

                  ```

              )EOF");
}

} // namespace netket

#endif
