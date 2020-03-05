//
// Created by Filippo Vicentini on 08/11/2019.
//

#include "Machine/DensityMatrices/py_abstract_density_matrix.hpp"

#include <complex>
#include <limits>
#include <vector>

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "Utils/messages.hpp"

#include "abstract_density_matrix.hpp"
#include "diagonal_density_matrix.hpp"
#include "ndm_spin_phase.hpp"

namespace py = pybind11;

namespace netket {

namespace {

void AddNdmSpinPhase(py::module &subm) {
  py::class_<NdmSpinPhase, AbstractDensityMatrix>(subm, "NdmSpinPhase", R"EOF(
          A positive semidefinite Neural Density Matrix (NDM) with real-valued parameters.
          In this case, two NDMs are taken to parameterize, respectively, phase
          and amplitude of the density matrix.
          This type of NDM has spin 1/2 hidden and ancilla units.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, int, int, int, int,
                    bool, bool, bool>(),
           py::arg("hilbert"), py::arg("n_hidden") = 0,
           py::arg("n_ancilla") = 0, py::arg("alpha") = 0, py::arg("beta") = 0,
           py::arg("use_visible_bias") = true,
           py::arg("use_hidden_bias") = true,
           py::arg("use_ancilla_bias") = true,
           R"EOF(
                   Constructs a new ``NdmSpinPhase`` machine:

                   Args:
                       hilbert: physical Hilbert space over which the density matrix acts.
                       n_hidden:  Number of hidden units.
                       n_ancilla: Number of ancilla units.
                       alpha: Hidden unit density.
                       beta:  Ancilla unit density.
                       use_visible_bias: If ``True`` then there would be a
                                        bias on the visible units.
                                        Default ``True``.
                       use_hidden_bias:  If ``True`` then there would be a
                                       bias on the visible units.
                                       Default ``True``.
                       use_ancilla_bias: If ``True`` then there would be a
                                       bias on the ancilla units.
                                       Default ``True``.

                   Examples:
                       A ``NdmSpinPhase`` machine with hidden unit density
                       alpha = 1 and ancilla unit density beta = 2 for a
                       one-dimensional L=9 spin-half system:

                       ```python
                       >>> from netket.machine import NdmSpinPhase
                       >>> from netket.hilbert import Spin
                       >>> from netket.graph import Hypercube
                       >>> g = Hypercube(length=9, n_dim=1)
                       >>> hi = Spin(s=0.5, total_sz=0.5, graph=g)
                       >>> ma = NdmSpinPhase(hilbert=hi,alpha=1, beta=2)
                       >>> print(ma.n_par)
                       540

                       ```
                   )EOF");
}

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

void AddAbstractDensityMatrix(py::module &subm) {
  py::class_<AbstractDensityMatrix, AbstractMachine, PyAbstractDensityMatrix>(
      subm, "DensityMatrix")
      .def_property_readonly(
          "hilbert_physical", &AbstractDensityMatrix::GetHilbertPhysical,
          R"EOF(netket.hilbert.Hilbert: The physical hilbert space object of the density matrix.)EOF")
      .def_property_readonly(
          "hilbert", &AbstractDensityMatrix::GetHilbert,
          R"EOF(netket.hilbert.Hilbert: The hilbert space object of the system.)EOF")
      .def_property_readonly(
          "n_visible_physical", &AbstractDensityMatrix::NvisiblePhysical,
          R"EOF(int: The number of inputs into the machine aka visible units in
            the case of Restricted Boltzmann Machines.)EOF")
      .def("to_matrix",
           [](AbstractDensityMatrix &self,
              bool normalize) -> AbstractDensityMatrix::MatrixType {
             const auto &hind = self.GetHilbertPhysical().GetIndex();
             AbstractMachine::MatrixType vals(hind.NStates(), hind.NStates());

             double maxlog = std::numeric_limits<double>::lowest();

             for (Index i = 0; i < hind.NStates(); i++) {
               auto v_r = hind.NumberToState(i);
               for (Index j = 0; j < hind.NStates(); j++) {
                 auto v_c = hind.NumberToState(j);

                 vals(i, j) = self.LogValSingle(v_r, v_c, {});
                 if (std::real(vals(i, j)) > maxlog) {
                   maxlog = std::real(vals(i, j));
                 }
               }
             }

             vals.array() -= maxlog;
             vals = vals.array().exp();

             if (normalize) {
               vals /= vals.trace();
             }
             return vals;
           },
           py::arg("normalize") = true,
           R"EOF(
                Returns a numpy matrix representation of the machine.
                The returned matrix has trace normalized to 1 if `normalize=True`
                Note that, in general, the size of the matrix is exponential
                in the number of quantum numbers, and this operation should thus
                only be performed for low-dimensional Hilbert spaces.

                This method requires an indexable Hilbert space.
              )EOF")
      .def("log_trace",
           [](AbstractDensityMatrix &self) -> double {
             const auto &hind = self.GetHilbertPhysical().GetIndex();
             AbstractMachine::MatrixType vals(hind.NStates(), hind.NStates());

             double maxlog = std::numeric_limits<double>::lowest();

             for (Index i = 0; i < hind.NStates(); i++) {
               auto v_r = hind.NumberToState(i);
               for (Index j = 0; j < hind.NStates(); j++) {
                 auto v_c = hind.NumberToState(j);

                 vals(i, j) = self.LogValSingle(v_r, v_c, {});
                 if (std::real(vals(i, j)) > maxlog) {
                   maxlog = std::real(vals(i, j));
                 }
               }
             }

             vals.array() -= maxlog;
             vals = vals.array().exp();

             return std::log(vals.trace().real()) + maxlog;
           },
           R"EOF(
                        Returns the log of the trace of the density matrix.

                        This method requires an indexable Hilbert space.
                      )EOF")
      .def("diagonal",
           [](AbstractDensityMatrix &self) -> DiagonalDensityMatrix {
             return DiagonalDensityMatrix(self);
           },
           R"EOF(
             Returns the wrapped machine sampling the diagonal of this density matrix.
             )EOF")
      .def("log_val",
           [](AbstractDensityMatrix &self, py::array_t<double> xr,
              py::array_t<double> xc) {
             // Ugly. Should use std::optional but does not work with pybind11
             // If there is only one input
             if (xc.ndim() == 0) {
               if (xr.ndim() == 1) {
                 auto input = xr.cast<Eigen::Ref<const VectorXd>>();
                 return py::cast(self.LogValSingle(input, any{}));
               } else if (xr.ndim() == 2) {
                 auto input = xr.cast<Eigen::Ref<const RowMatrix<double>>>();
                 return py::cast(self.AbstractMachine::LogVal(input, any{}));
               } else if (xr.ndim() == 3) {
                 auto input = Eigen::Map<const RowMatrix<double>>{
                     xr.data(), xr.shape(0) * xr.shape(1), xr.shape(2)};
                 py::array_t<Complex> result =
                     py::cast(self.AbstractMachine::LogVal(input, any{}));
                 result.resize({xr.shape(0), xr.shape(1),
                                static_cast<pybind11::ssize_t>(self.Npar())});
                 return py::object(result);
               } else {
                 throw InvalidInputError{"Invalid input dimensions"};
               }
             }

             if (xr.ndim() != xc.ndim()) {
               throw InvalidInputError{
                   "Row and columns must have same dimensions"};
             }
             if (xr.ndim() == 1) {
               auto input_r = xr.cast<Eigen::Ref<const VectorXd>>();
               auto input_c = xc.cast<Eigen::Ref<const VectorXd>>();
               return py::cast(self.LogValSingle(input_r, input_c));
             } else if (xr.ndim() == 2) {
               auto input_r = xr.cast<Eigen::Ref<const RowMatrix<double>>>();
               auto input_c = xc.cast<Eigen::Ref<const RowMatrix<double>>>();
               return py::cast(self.LogVal(input_r, input_c, any{}));
             } else if (xr.ndim() == 3) {
               auto input_r = Eigen::Map<const RowMatrix<double>>{
                   xr.data(), xr.shape(0) * xr.shape(1), xr.shape(2)};
               auto input_c = Eigen::Map<const RowMatrix<double>>{
                   xc.data(), xc.shape(0) * xc.shape(1), xc.shape(2)};
               py::array_t<Complex> result =
                   py::cast(self.LogVal(input_r, input_c, any{}));
               result.resize({xr.shape(0), xr.shape(1),
                              static_cast<pybind11::ssize_t>(self.Npar())});
               return py::object(result);
             }
             { throw InvalidInputError{"Invalid input dimension"}; }
           },
           py::arg("x_row"), py::arg("x_col") = nullptr,
           R"EOF(
                  Member function to obtain log value of machine given an input
                 vector.

                 Args:
                     vr: Input vector to machine.
                     vc: Input vector to machine.
                )EOF")
      .def("der_log",
           [](AbstractDensityMatrix &self, py::array_t<double> xr,
              py::array_t<double> xc) {
             // Ugly. Should use std::optional but does not work with pybind11
             if (xc.ndim() == 0) {
               if (xr.ndim() == 1) {
                 auto input = xr.cast<Eigen::Ref<const VectorXd>>();
                 return py::cast(self.DerLogSingle(input, any{}));
               } else if (xr.ndim() == 2) {
                 auto input = xr.cast<Eigen::Ref<const RowMatrix<double>>>();
                 return py::cast(self.AbstractMachine::DerLog(input, any{}));
               } else if (xr.ndim() == 3) {
                 auto input = Eigen::Map<const RowMatrix<double>>{
                     xr.data(), xr.shape(0) * xr.shape(1), xr.shape(2)};
                 py::array_t<Complex> result =
                     py::cast(self.AbstractMachine::DerLog(input, any{}));
                 result.resize({xr.shape(0), xr.shape(1),
                                static_cast<pybind11::ssize_t>(self.Npar())});
                 return py::object(result);
               } else {
                 throw InvalidInputError{"Invalid input dimensions"};
               }
             }

             if (xr.ndim() != xc.ndim()) {
               throw InvalidInputError{
                   "Row and columns must have same dimensions"};
             }
             if (xr.ndim() == 1) {
               auto input_r = xr.cast<Eigen::Ref<const VectorXd>>();
               auto input_c = xc.cast<Eigen::Ref<const VectorXd>>();
               return py::cast(self.DerLogSingle(input_r, input_c));
             } else if (xr.ndim() == 2) {
               auto input_r = xr.cast<Eigen::Ref<const RowMatrix<double>>>();
               auto input_c = xc.cast<Eigen::Ref<const RowMatrix<double>>>();
               return py::cast(self.DerLog(input_r, input_c, any{}));
             } else if (xr.ndim() == 3) {
               auto input_r = Eigen::Map<const RowMatrix<double>>{
                   xr.data(), xr.shape(0) * xr.shape(1), xr.shape(2)};
               auto input_c = Eigen::Map<const RowMatrix<double>>{
                   xc.data(), xc.shape(0) * xc.shape(1), xc.shape(2)};
               py::array_t<Complex> result =
                   py::cast(self.DerLog(input_r, input_c, any{}));
               result.resize({xr.shape(0), xr.shape(1),
                              static_cast<pybind11::ssize_t>(self.Npar())});
               return py::object(result);
             }
             { throw InvalidInputError{"Invalid input dimension"}; }
           },
           py::arg("x_row"), py::arg("x_col") = nullptr,
           R"EOF(
                 Member function to obtain the derivatives of log value of
                 machine given an input wrt the machine's parameters.

                 Args:
                     vr: Input row vector to machine
                     vc: Input column vector to machine

                 )EOF");
}
}  // namespace

void AddDensityMatrixModule(py::module subm) {
  // auto subm = m.def_submodule("densitymatrix");

  AddAbstractDensityMatrix(subm);
  AddNdmSpinPhase(subm);
  AddDiagonalDensityMatrix(subm);
}

}  // namespace netket
