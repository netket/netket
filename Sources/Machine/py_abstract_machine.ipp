// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

//#include "Machine/py_abstract_machine.hpp"

#include <cstdio>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "Utils/messages.hpp"

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

template <class AMB>
int PyAbstractMachine<AMB>::Npar() const {
  return detail::ShouldNotThrow([this]() {
    PYBIND11_OVERLOAD_PURE_NAME(int,             /* Return type */
                                AbstractMachine, /* Parent class */
                                "_n_par", /* Name of the function in Python */
                                Npar,     /* Name of function in C++ */
    );
  });
}

template <class AMB>
bool PyAbstractMachine<AMB>::IsHolomorphic() const noexcept {
  return detail::ShouldNotThrow([this]() {
    PYBIND11_OVERLOAD_PURE_NAME(
        bool,              /* Return type */
        AMB,               /* Parent class */
        "_is_holomorphic", /* Name of the function in Python */
        IsHolomorphic,     /* Name of function in C++ */
    );
  });
}

template <class AMB>
typename AMB::VectorType PyAbstractMachine<AMB>::GetParameters() {
  PYBIND11_OVERLOAD_PURE_NAME(
      VectorType,        /* Return type */
      AMB,               /* Parent class */
      "_get_parameters", /* Name of the function in Python */
      GetParameters,     /* Name of function in C++ */
  );
}

template <class AMB>
void PyAbstractMachine<AMB>::SetParameters(VectorConstRefType pars) {
  PYBIND11_OVERLOAD_PURE_NAME(
      void,              /* Return type */
      AMB,               /* Parent class */
      "_set_parameters", /* Name of the function in Python */
      SetParameters,     /* Name of function in C++ */
      pars);
}

template <class AMB>
void PyAbstractMachine<AMB>::LogVal(Eigen::Ref<const RowMatrix<double>> v,
                                    Eigen::Ref<Eigen::VectorXcd> out,
                                    const any & /*unused*/) {
  PYBIND11_OVERLOAD_PURE_NAME(void,      /* Return type */
                              AMB,       /* Parent class */
                              "log_val", /* Name of the function in Python */
                              LogVal,    /* Name of function in C++ */
                              v, out);
}

template <class AMB>
Complex PyAbstractMachine<AMB>::LogValSingle(VisibleConstType v,
                                             const any &cache) {
  Complex data;
  auto out = Eigen::Map<Eigen::VectorXcd>(&data, 1);
  LogVal(v.transpose(), out, cache);
  return data;
}

template <class AMB>
any PyAbstractMachine<AMB>::InitLookup(VisibleConstType /*unused*/) {
  return {};
}

template <class AMB>
void PyAbstractMachine<AMB>::UpdateLookup(
    VisibleConstType /*unused*/, const std::vector<int> & /*unused*/,
    const std::vector<double> & /*unused*/, any & /*unused*/) {}

template <class AMB>
void PyAbstractMachine<AMB>::DerLog(Eigen::Ref<const RowMatrix<double>> v,
                                    Eigen::Ref<RowMatrix<Complex>> out,
                                    const any & /*unused*/) {
  PYBIND11_OVERLOAD_PURE_NAME(void,      /* Return type */
                              AMB,       /* Parent class */
                              "der_log", /* Name of the function in Python */
                              DerLog,    /* Name of function in C++ */
                              v, out);
}

template <class AMB>
PyObject *PyAbstractMachine<AMB>::StateDict() {
  return [this]() {
    PYBIND11_OVERLOAD_PURE_NAME(
        pybind11::object, /* Return type */
        AMB,              /* Parent class */
        "state_dict",     /* Name of the function in Python */
        StateDict,        /* Name of function in C++ */
                          /*Arguments*/
    );
  }()
             .release()
             .ptr();
}

template <class AMB>
void PyAbstractMachine<AMB>::Save(const std::string &filename) const {
  PYBIND11_OVERLOAD_NAME(void,    /* Return type */
                         AMB,     /* Parent class */
                         "save",  /* Name of the function in Python */
                         Save,    /* Name of function in C++ */
                         filename /*Arguments*/
  );
}

template <class AMB>
void PyAbstractMachine<AMB>::Load(const std::string &filename) {
  PYBIND11_OVERLOAD_NAME(void,    /* Return type */
                         AMB,     /* Parent class */
                         "load",  /* Name of the function in Python */
                         Load,    /* Name of function in C++ */
                         filename /*Arguments*/
  );
}

}  // namespace netket
