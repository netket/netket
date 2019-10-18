#include "py_variational_montecarlo.hpp"

#include <complex>
#include <vector>

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "Utils/pybind_helpers.hpp"
#include "variational_montecarlo.hpp"

namespace py = pybind11;

namespace netket {

void AddVariationalMonteCarloModule(PyObject *module) {
  py::module m{py::reinterpret_borrow<py::object>(module)};
  auto m_vmc = m.def_submodule("variational");

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
      .def("get_observable_stats",
           [](VariationalMonteCarlo &self) {
             self.ComputeObservables();
             return self.GetObservableStats();
           },
           R"EOF(
        Calculate and return the value of the operators stored as observables.

        )EOF")
      .def_property_readonly("vmc_data", &VariationalMonteCarlo::GetVmcData)
      .def_property_readonly("sr", &VariationalMonteCarlo::GetSR);
}

}  // namespace netket
