//
// Created by Filippo Vicentini on 08/11/2019.
//

#include "Machine/DensityMatrices/py_abstract_density_matrix.hpp"

#include <cstdio>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace netket {

namespace detail {
namespace {
bool ShouldIDoIO() noexcept {
  auto rank = 0;
  auto const status = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (status == MPI_SUCCESS) {
    return rank == 0;
  }
  std::fprintf(stderr,
               "[NetKet] MPI_Comm_rank failed: doing I/O on all processes.\n");
  return true;
}

template <class Function, class... Args>
auto ShouldNotThrow(Function &&function, Args &&... args) noexcept
-> decltype(std::declval<Function &&>()(std::declval<Args &&>()...)) {
  try {
    return std::forward<Function>(function)(std::forward<Args>(args)...);
  } catch (std::exception &e) {
    if (ShouldIDoIO()) {
      std::fprintf(
          stderr,
          "[NetKet] Fatal error: exception was thrown in a `noexcept` context\n"
          "[NetKet]        Info: %s\n"
          "[NetKet]\n"
          "[NetKet] This is a bug. Please, be so kind to open an issue at\n"
          "[NetKet]             https://github.com/netket/netket/issues\n"
          "[NetKet]\n"
          "[NetKet] Aborting...\n",
          e.what());
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
    std::abort();  // This call is unreachable and is here just to tell the
    // compiler that this branch never returns (MPI_Abort is not
    // marked [[noreturn]]).
  } catch (...) {
    if (ShouldIDoIO()) {
      std::fprintf(
          stderr,
          "[NetKet] Fatal error: exception was thrown in a `noexcept` context\n"
          "[NetKet] Aborting...\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
    std::abort();  // This call is unreachable and is here just to tell the
    // compiler that this branch never returns (MPI_Abort is not
    // marked [[noreturn]]).
  }
}
}  // namespace
}  // namespace detail



Complex PyAbstractDensityMatrix::LogValSingle(VisibleConstType v,
                                              const any& cache) {
  Complex data;
  std::cout << "logvalsingle py 1" << v << std::endl << std::flush;
  auto out = Eigen::Map<Eigen::VectorXcd>(&data, 1);
  AbstractMachine::LogVal(v.transpose(), out, cache);
  return data;
}

void PyAbstractDensityMatrix::LogVal(Eigen::Ref<const RowMatrix<double>> vr,
                                     Eigen::Ref<const RowMatrix<double>> vc,
                                     Eigen::Ref<VectorXcd> out,
                                     const linb::any&) {
  PYBIND11_OVERLOAD_PURE_NAME(void,                  /* Return type */
                              AbstractDensityMatrix, /* Parent class */
                              "log_val", /* Name of the function in Python */
                              LogVal,    /* Name of function in C++ */
                              vr, vc, out);
}

Complex PyAbstractDensityMatrix::LogValSingle(VisibleConstType vr,
                                              VisibleConstType vc,
                                              const any& cache) {
  std::cout << "pydispatch douearg - single in" << std::endl;
  std::cout << vr << std::endl;
  std::cout << vc << std::endl;

  Complex data;
  auto out = Eigen::Map<Eigen::VectorXcd>(&data, 1);
  LogVal(vr.transpose(), vc.transpose(), out, cache);
  return data;
}

void PyAbstractDensityMatrix::DerLog(Eigen::Ref<const RowMatrix<double>> vr,
                                     Eigen::Ref<const RowMatrix<double>> vc,
                                     Eigen::Ref<RowMatrix<Complex>> out,
                                     const linb::any& cache) {
  PYBIND11_OVERLOAD_PURE_NAME(void,                  /* Return type */
                              AbstractDensityMatrix, /* Parent class */
                              "der_log", /* Name of the function in Python */
                              DerLog,    /* Name of function in C++ */
                              vr, vc, out);
}

PyAbstractDensityMatrix::VectorType PyAbstractDensityMatrix::DerLogSingle(
    VisibleConstType vr, VisibleConstType vc, const any& cache) {
  Eigen::VectorXcd out(Npar());
  DerLog(vr.transpose(), vc.transpose(),
         Eigen::Map<RowMatrix<Complex>>{out.data(), 1, out.size()}, cache);
  return out;
}

}  // namespace netket
