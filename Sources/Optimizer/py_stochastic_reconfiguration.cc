#include "Optimizer/py_stochastic_reconfiguration.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "Utils/exceptions.hpp"
#include "Utils/pybind_helpers.hpp"
#include "common_types.hpp"

namespace netket {

namespace py = pybind11;

void AddSR(py::module& m) {
  py::class_<SR>(m, "SR", "Performs stochastic reconfiguration (SR) updates")
      .def(py::init([](const std::string& solver_name, double diag_shift,
                       bool use_iterative, bool is_holomorphic) {
             const auto solver = SR::SolverFromString(solver_name);
             NETKET_CHECK(solver.has_value(), InvalidInputError,
                          "Invalid LSQ solver \"" << solver_name
                                                  << "\" specified for SR");
             return SR(solver.value(), diag_shift, use_iterative,
                       is_holomorphic);
           }),
           py::arg("lsq_solver") = "LLT", py::arg("diag_shift") = 0.01,
           py::arg("use_iterative") = false, py::arg("is_holomorphic") = true)
      .def("compute_update", &SR::ComputeUpdate, py::arg("Oks").noconvert(),
           py::arg("grad").noconvert(), py::arg("out").noconvert(),
           R"EOF(
            Solves the SR flow equation for the parameter update ẋ.

            The SR update is computed by solving the linear equation
               Sẋ = f
            where S is the covariance matrix of the partial derivatives
            O_i(v_j) = ∂/∂x_i log Ψ(v_j) and f is a generalized force (the loss
            gradient).

            Args:
                Oks: The matrix 𝕆 of centered log-derivatives,
                   𝕆_ij = O_i(v_j) - ⟨O_i⟩.
                grad: The vector of forces f.
                out: Output array for the update ẋ.
          )EOF")
      .def_property("store_rank_enabled", &SR::StoreRankEnabled,
                    &SR::SetStoreRank)
      .def_property(
          "scale_invariant_regularization_enabled",
          &SR::ScaleInvariantRegularizationEnabled,
          &SR::SetScaleInvariantRegularization,
          R"EOF(bool: Whether to use the scale-invariant regularization as described by
               Becca and Sorella (2017), pp. 143-144.
               https://doi.org/10.1017/9781316417041")EOF")
      .def_property("store_covariance_matrix_enabled",
                    &SR::StoreFullSMatrixEnabled, &SR::SetStoreFullSMatrix)
      .def_property_readonly("last_rank", &SR::LastRank)
      .def_property_readonly("last_covariance_matrix", &SR::LastSMatrix)
      .def("info", &SR::LongDesc, py::arg("depth") = 0)
      .def("__repr__", &SR::ShortDesc);
}

}  // namespace netket
