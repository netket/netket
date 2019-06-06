// Copyright 2018 The Simons Foundation, Inc. - All
// Rights Reserved.
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

#ifndef NETKET_PY_DENSITY_MATRIX_HPP
#define NETKET_PY_DENSITY_MATRIX_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "abstract_density_matrix.hpp"
#include "py_diagonal_density_matrix.hpp"
#include "py_ndm_spin_phase.hpp"

namespace py = pybind11;

namespace netket {
void AddDensityMatrixModule(py::module &subm) {
  py::class_<AbstractDensityMatrix, AbstractMachine>(subm, "DensityMatrix")
      .def_property_readonly(
          "hilbert_physical", &AbstractDensityMatrix::GetHilbertPhysical,
          R"EOF(netket.hilbert.Hilbert: The physical hilbert space object of the density matrix.)EOF")
      .def(
          "to_matrix",
          [](AbstractDensityMatrix &self) -> AbstractDensityMatrix::MatrixType {
            const auto &hind = self.GetHilbertPhysical().GetIndex();
            AbstractMachine::MatrixType vals(hind.NStates(), hind.NStates());

            double maxlog = std::numeric_limits<double>::lowest();

            for (Index i = 0; i < hind.NStates(); i++) {
              auto v_r = hind.NumberToState(i);
              for (Index j = 0; j < hind.NStates(); j++) {
                auto v_c = hind.NumberToState(j);

                Eigen::VectorXd v(v_r.size() * 2);
                v << v_r, v_c;

                vals(i, j) = self.LogVal(v);
                if (std::real(vals(i, j)) > maxlog) {
                  maxlog = std::real(vals(i, j));
                }
              }
            }

            vals.array() -= maxlog;
            vals = vals.array().exp();

            vals /= vals.trace();
            return vals;
          },
          R"EOF( a
                Returns a numpy matrix representation of the machine.
                The returned matrix has trace normalized to 1,
                Note that, in general, the size of the matrix is exponential
                in the number of quantum numbers, and this operation should thus
                only be performed for low-dimensional Hilbert spaces.

                This method requires an indexable Hilbert space.
              )EOF");

  AddDiagonalDensityMatrix(subm);

  AddNdmSpinPhase(subm);
}
}  // namespace netket

#endif  // NETKET_PY_DENSITY_MATRIX_HPP
