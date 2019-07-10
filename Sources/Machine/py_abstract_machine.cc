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

#include "Machine/py_abstract_machine.hpp"

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

int PyAbstractMachine::Npar() const {
  PYBIND11_OVERLOAD_PURE_NAME(
      int,                  /* Return type */
      AbstractMachine,      /* Parent class */
      "_number_parameters", /* Name of the function in Python */
      Npar,                 /* Name of function in C++ */
  );
}

int PyAbstractMachine::Nvisible() const {
  PYBIND11_OVERLOAD_PURE_NAME(
      int,               /* Return type */
      AbstractMachine,   /* Parent class */
      "_number_visible", /* Name of the function in Python */
      Nvisible,          /* Name of function in C++ */
  );
}

bool PyAbstractMachine::IsHolomorphic() const noexcept {
  return detail::ShouldNotThrow([this]() {
    PYBIND11_OVERLOAD_PURE_NAME(
        bool,              /* Return type */
        AbstractMachine,   /* Parent class */
        "_is_holomorphic", /* Name of the function in Python */
        IsHolomorphic,     /* Name of function in C++ */
    );
  });
}

PyAbstractMachine::VectorType PyAbstractMachine::GetParameters() {
  PYBIND11_OVERLOAD_PURE_NAME(
      VectorType,        /* Return type */
      AbstractMachine,   /* Parent class */
      "_get_parameters", /* Name of the function in Python */
      GetParameters,     /* Name of function in C++ */
  );
}

void PyAbstractMachine::SetParameters(VectorConstRefType pars) {
  PYBIND11_OVERLOAD_PURE_NAME(
      void,              /* Return type */
      AbstractMachine,   /* Parent class */
      "_set_parameters", /* Name of the function in Python */
      SetParameters,     /* Name of function in C++ */
      pars);
}

Complex PyAbstractMachine::LogValSingle(VisibleConstType v,
                                        const any & /*unused*/) {
  PYBIND11_OVERLOAD_PURE_NAME(
      Complex,          /* Return type */
      AbstractMachine,  /* Parent class */
      "log_val_single", /* Name of the function in Python */
      LogValSingle,     /* Name of function in C++ */
      v);
}

any PyAbstractMachine::InitLookup(VisibleConstType /*unused*/) { return {}; }

void PyAbstractMachine::UpdateLookup(VisibleConstType /*unused*/,
                                     const std::vector<int> & /*unused*/,
                                     const std::vector<double> & /*unused*/,
                                     any & /*unused*/) {}

Complex PyAbstractMachine::LogValDiff(VisibleConstType old_v,
                                      const std::vector<int> &to_change,
                                      const std::vector<double> &new_conf) {
  auto const old_value = LogValSingle(old_v, {});
  VisibleType new_v{old_v};
  GetHilbert().UpdateConf(new_v, to_change, new_conf);
  auto const new_value = LogValSingle(new_v, {});
  return new_value - old_value;
}

PyAbstractMachine::VectorType PyAbstractMachine::LogValDiff(
    VisibleConstType old_v, const std::vector<std::vector<int>> &to_change,
    const std::vector<std::vector<double>> &new_conf) {
  assert(to_change.size() == new_conf.size());
  VectorType result(to_change.size());
  for (auto i = size_t{0}; i < to_change.size(); ++i) {
    result(i) = LogValDiff(old_v, to_change[i], new_conf[i]);
  }
  return result;
}

Complex PyAbstractMachine::LogValDiff(VisibleConstType v,
                                      const std::vector<int> &to_change,
                                      const std::vector<double> &new_conf,
                                      const any & /*unused*/) {
  return LogValDiff(v, to_change, new_conf);
}

PyAbstractMachine::VectorType PyAbstractMachine::DerLogSingle(
    VisibleConstType v, const any & /*lt*/) {
  PYBIND11_OVERLOAD_PURE_NAME(
      VectorType,       /* Return type */
      AbstractMachine,  /* Parent class */
      "der_log_single", /* Name of the function in Python */
      DerLogSingle,     /* Name of function in C++ */
      v);
}

PyAbstractMachine::VectorType PyAbstractMachine::DerLogChanged(
    VisibleConstType old_v, const std::vector<int> &to_change,
    const std::vector<double> &new_conf) {
  VisibleType new_v{old_v};
  GetHilbert().UpdateConf(new_v, to_change, new_conf);
  return DerLogSingle(new_v, {});
}

void PyAbstractMachine::Save(const std::string &filename) const {
  PYBIND11_OVERLOAD_PURE_NAME(void,            /* Return type */
                              AbstractMachine, /* Parent class */
                              "save",  /* Name of the function in Python */
                              Save,    /* Name of function in C++ */
                              filename /*Arguments*/
  );
}

void PyAbstractMachine::Load(const std::string &filename) {
  PYBIND11_OVERLOAD_PURE_NAME(void,            /* Return type */
                              AbstractMachine, /* Parent class */
                              "load",  /* Name of the function in Python */
                              Load,    /* Name of function in C++ */
                              filename /*Arguments*/
  );
}

}  // namespace netket
