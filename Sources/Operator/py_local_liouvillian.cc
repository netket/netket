//
// Created by Filippo Vicentini on 08/11/2019.
//

#include "py_local_liouvillian.hpp"
#include <pybind11/stl.h>

namespace netket {
void AddLocalSuperOperatorModule(py::module &subm) {
  py::class_<LocalLiouvillian, AbstractOperator,
             std::shared_ptr<LocalLiouvillian>>(
      subm, "LocalLiouvillian", R"EOF(A custom local super-operator.)EOF")
      .def(py::init<const LocalOperator &,
               const std::vector<const LocalOperator> &>(),
           py::keep_alive<1, 2>(), py::arg("hamiltonian"),
           py::arg("jump_ops"),
           R"EOF(
           Constructs a new ``LocalLiouvillian`` given a Hamilotnian and a
           list of ``Jump Operators``
           ```
           )EOF")
      .def_property_readonly(
          "jump_ops", &LocalLiouvillian::GetJumpOperators,
          R"EOF(list[list]: A list of the local matrices.)EOF")
      .def("add_jump_op", &LocalLiouvillian::AddJumpOperator,
           R"EOF(jump_op: add jump op.)EOF")
      .def("get_effective_hamiltonian",
           &LocalLiouvillian::GetEffectiveHamiltonian);
}
}  // namespace netket
