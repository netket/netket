//
// Created by Filippo Vicentini on 08/11/2019.
//

#include "Machine/DensityMatrices/py_abstract_density_matrix.hpp"

#include <cstdio>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace netket {

template <class ADM>
Complex PyAbstractDensityMatrix<ADM>::LogValSingle(VisibleConstType v,
                                              const any& cache) {
  Complex data;
  auto out = Eigen::Map<Eigen::VectorXcd>(&data, 1);
  ADM::AbstractMachine::LogVal(v.transpose(), out, cache);
  return data;
}

template <class ADM>
void PyAbstractDensityMatrix<ADM>::LogVal(Eigen::Ref<const RowMatrix<double>> vr,
                                     Eigen::Ref<const RowMatrix<double>> vc,
                                     Eigen::Ref<VectorXcd> out,
                                     const linb::any& cache) {
  PYBIND11_OVERLOAD_PURE_NAME(void,                  /* Return type */
                              ADM, /* Parent class */
                              "log_val", /* Name of the function in Python */
                              LogVal,    /* Name of function in C++ */
                              vr, vc, out, cache);
}

template <class ADM>
Complex PyAbstractDensityMatrix<ADM>::LogValSingle(VisibleConstType vr,
                                              VisibleConstType vc,
                                              const any& cache) {
  Complex data;
  auto out = Eigen::Map<Eigen::VectorXcd>(&data, 1);
  LogVal(vr.transpose(), vc.transpose(), out, cache);
  return data;
}

template <class ADM>
void PyAbstractDensityMatrix<ADM>::DerLog(Eigen::Ref<const RowMatrix<double>> vr,
                                     Eigen::Ref<const RowMatrix<double>> vc,
                                     Eigen::Ref<RowMatrix<Complex>> out,
                                     const linb::any& cache) {
  PYBIND11_OVERLOAD_PURE_NAME(void,                  /* Return type */
                              ADM, /* Parent class */
                              "der_log", /* Name of the function in Python */
                              DerLog,    /* Name of function in C++ */
                              vr, vc, out, cache);
}

/*
template <class ADM>
PyAbstractDensityMatrix::VectorType PyAbstractDensityMatrix::DerLogSingle(
    VisibleConstType vr, VisibleConstType vc, const any& cache) {
  Eigen::VectorXcd out(Npar());
  DerLog(vr.transpose(), vc.transpose(),
         Eigen::Map<RowMatrix<Complex>>{out.data(), 1, out.size()}, cache);
  return out;
}*/

}  // namespace netket
