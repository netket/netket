//
// Created by Filippo Vicentini on 08/11/2019.
//

#include "py_local_lindbladian.hpp"

namespace netket {
void AddLocalSuperOperatorModule(py::module &subm) {
  py::class_<LocalLindbladian, AbstractOperator,
             std::shared_ptr<LocalLindbladian>>(
      subm, "LocalLindbladian", R"EOF(A custom local super-operator.)EOF")
      .def(py::init<const LocalOperator &>(), py::keep_alive<1, 2>(),
           py::arg("hamiltonian"),
           R"EOF(
           Constructs a new ``LocalLindbladian`` given a Hamilotnian and a
           list of ``Jump Operators``
           ```
           )EOF")
      .def_property_readonly(
          "jump_ops", &LocalLindbladian::GetJumpOperators,
          R"EOF(list[list]: A list of the local matrices.)EOF")
      .def("add_jump_op", &LocalLindbladian::AddJumpOperator,
           R"EOF(jump_op: add jump op.)EOF")
      .def("get_effective_hamiltonian", &LocalLindbladian::GetEffectiveHamiltonian);
}
}  // namespace netket
