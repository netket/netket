//
// Created by Filippo Vicentini on 2019-06-05.
//

#ifndef NETKET_PY_DIAGONAL_DENSITY_MATRIX_HPP
#define NETKET_PY_DIAGONAL_DENSITY_MATRIX_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "diagonal_density_matrix.hpp"

namespace py = pybind11;

namespace netket {

void AddDiagonalDensityMatrix(py::module &subm) {
  py::class_<DiagonalDensityMatrix, AbstractMachine>(
      subm, "DiagonalDensityMatrix", R"EOF(
  A Machine sampling the diagonal of a density matrix.)EOF")
      .def(py::init<AbstractDensityMatrix &>(), py::keep_alive<1, 2>(),
           py::arg("dm"), R"EOF(

               Constructs a new ``DiagonalDensityMatrix`` machine sampling the
               diagonal of the provided density matrix.

               Args:
                    dm: the density matrix.
)EOF");
}

}  // namespace netket
#endif  // NETKET_PY_DIAGONAL_DENSITY_MATRIX_HPP
