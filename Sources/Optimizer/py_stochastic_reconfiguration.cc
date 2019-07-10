#include "Optimizer/py_stochastic_reconfiguration.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "common_types.hpp"

namespace py = pybind11;

namespace netket {
namespace detail {
namespace {
void AddSR(py::module m) {
  py::class_<SR>(m, "SR")
      .def(py::init<double, bool, bool, bool>(), py::arg{"diag_shift"} = 0.01,
           py::arg{"use_iterative"} = false, py::arg{"use_cholesky"} = true,
           py::arg{"is_holomorphic"} = true)
      .def_readwrite("diag_shift", &SR::diag_shift)
      .def_readwrite("use_iterative", &SR::use_iterative)
      .def_readwrite("use_cholesky", &SR::use_cholesky)
      .def_readwrite("is_holomorphic", &SR::is_holomorphic)
      .def(
          "compute_update",
          [](SR& self, Eigen::Ref<const MatrixXcd> Oks,
             Eigen::Ref<const VectorXcd> grad,
             Eigen::Ref<VectorXcd> out) { self.ComputeUpdate(Oks, grad, out); },
          py::arg{"Oks"}.noconvert(), py::arg{"grad"}.noconvert(),
          py::arg{"out"}.noconvert());
}
}  // namespace
}  // namespace detail

void AddSR(PyObject* m) {
  detail::AddSR(py::module{py::reinterpret_borrow<py::object>(m)});
}

}  // namespace netket
