// Copyright 2018-2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#include "Machine/abstract_machine.hpp"

#include <mpi.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <fstream>

#include "Utils/exceptions.hpp"
#include "Utils/messages.hpp"
#include "Utils/mpi_interface.hpp"
#include "Utils/pybind_helpers.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

void AbstractMachine::InitRandomPars(double sigma,
                                     nonstd::optional<unsigned> seed,
                                     default_random_engine *given_gen) {
  VectorType parameters(Npar());
  const auto comm = MPI_COMM_WORLD;
  constexpr auto root = 0;
  int rank;
  MPI_Comm_rank(comm, &rank);

  default_random_engine generator;
  if (given_gen != nullptr) {
    generator = *given_gen;
    if (seed.has_value()) {
      InfoMessage() << "Warning: the given seed does not have any effect"
                    << std::endl;
    }
  } else {
    if (seed.has_value()) {
      generator = default_random_engine(*seed);
    } else {
      generator = GetRandomEngine();
    }
  }

  if (rank == root) {
    std::generate(parameters.data(), parameters.data() + parameters.size(),
                  [&generator, sigma]() {
                    std::normal_distribution<double> dist{0.0, sigma};
                    return Complex{dist(generator), dist(generator)};
                  });
  }
  auto status = MPI_Bcast(parameters.data(), parameters.size(),
                          MPI_DOUBLE_COMPLEX, root, comm);
  if (status != MPI_SUCCESS) throw MPIError{status, "MPI_Bcast"};
  SetParameters(parameters);
}

void AbstractMachine::LogVal(Eigen::Ref<const RowMatrix<double>> v,
                             Eigen::Ref<VectorType> out,
                             const any & /*unused*/) {
  CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "out", out.size(), v.rows());
  for (auto i = Index{0}; i < v.rows(); ++i) {
    out(i) = LogValSingle(v.row(i));
  }
}

AbstractMachine::VectorType AbstractMachine::LogVal(
    Eigen::Ref<const RowMatrix<double>> v, const any &cache) {
  VectorType out(v.rows());
  LogVal(v, out, cache);
  return out;
}

void AbstractMachine::DerLog(Eigen::Ref<const RowMatrix<double>> v,
                             Eigen::Ref<RowMatrix<Complex>> out,
                             const any & /*cache*/) {
  CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "out", {out.rows(), out.cols()}, {v.rows(), Npar()});
  for (auto i = Index{0}; i < v.rows(); ++i) {
    out.row(i) = DerLogSingle(v.row(i));
  }
}

RowMatrix<Complex> AbstractMachine::DerLog(
    Eigen::Ref<const RowMatrix<double>> v, const any &cache) {
  RowMatrix<Complex> out(v.rows(), Npar());
  DerLog(v, out, cache);
  return out;
}

AbstractMachine::VectorType AbstractMachine::DerLogChanged(
    VisibleConstType v, const std::vector<int> &tochange,
    const std::vector<double> &newconf) {
  VisibleType vp(v);
  hilbert_->UpdateConf(vp, tochange, newconf);
  return DerLogSingle(vp);
}

void AbstractMachine::DerLogChanged(
    VisibleConstType v, const std::vector<std::vector<int>> &tochange,
    const std::vector<std::vector<double>> &newconf,
    Eigen::Ref<RowMatrix<Complex>> output) {
  DerLogDiff(v, tochange, newconf, output, false);
}

RowMatrix<Complex> AbstractMachine::DerLogDiff(
    VisibleConstType v, const std::vector<std::vector<int>> &tochange,
    const std::vector<std::vector<double>> &newconf) {
  auto output = RowMatrix<Complex>(newconf.size(), Npar());
  DerLogDiff(v, tochange, newconf, output);
  return output;
}

void AbstractMachine::DerLogDiff(
    VisibleConstType v, const std::vector<std::vector<int>> &tochange,
    const std::vector<std::vector<double>> &newconf,
    Eigen::Ref<RowMatrix<Complex>> output, bool subtract_logv) {
  CheckShape(__FUNCTION__, "out", {output.rows(), output.cols()},
             {static_cast<Index>(newconf.size()), Npar()});

  RowMatrix<double> input(static_cast<Index>(tochange.size()), v.size());
  input = v.transpose().colwise().replicate(input.rows());

  nonstd::optional<Index> log_val_single_ind;

  for (auto i = Index{0}; i < input.rows(); ++i) {
    GetHilbert().UpdateConf(input.row(i), tochange[static_cast<size_t>(i)],
                            newconf[static_cast<size_t>(i)]);

    // One of the states is equal to v ?
    if (tochange[static_cast<size_t>(i)].size() == 0) {
      log_val_single_ind = i;
    }
  }

  // auto
  DerLog(input, output, any{});

  if (subtract_logv) {
    if (log_val_single_ind.has_value()) {
      output = output.rowwise() - output.row(log_val_single_ind.value());
    } else {
      output = output.rowwise() - DerLogSingle(v, any{}).transpose();
    }
  }
}

AbstractMachine::VectorType AbstractMachine::LogValDiff(
    VisibleConstType v, const std::vector<std::vector<int>> &tochange,
    const std::vector<std::vector<double>> &newconf) {
  Eigen::VectorXcd output(static_cast<Index>(tochange.size()));
  LogValDiff(v, tochange, newconf, output);
  return output;
}

void AbstractMachine::LogValDiff(
    VisibleConstType v, const std::vector<std::vector<int>> &tochange,
    const std::vector<std::vector<double>> &newconf,
    Eigen::Ref<Eigen::VectorXcd> output) {
  RowMatrix<double> input(static_cast<Index>(tochange.size()), v.size());
  input = v.transpose().colwise().replicate(input.rows());

  CheckShape(__FUNCTION__, "out", output.rows(),
             static_cast<Index>(newconf.size()));

  nonstd::optional<Index> log_val_single_ind;

  for (auto i = Index{0}; i < input.rows(); ++i) {
    GetHilbert().UpdateConf(input.row(i), tochange[static_cast<size_t>(i)],
                            newconf[static_cast<size_t>(i)]);

    // One of the states is equal to v ?
    if (tochange[static_cast<size_t>(i)].size() == 0) {
      log_val_single_ind = i;
    }
  }
  output.resize(input.rows());
  LogVal(input, output, any{});

  if (log_val_single_ind.has_value()) {
    output.array() -= output(log_val_single_ind.value());
  } else {
    output.array() -= LogValSingle(v, any{});
  }
}

any AbstractMachine::InitLookup(VisibleConstType /*v*/) { return any{}; }

void AbstractMachine::UpdateLookup(VisibleConstType, const std::vector<int> &,
                                   const std::vector<double> &, any &) {}

PyObject *AbstractMachine::StateDict() const {
  auto state = pybind11::reinterpret_steal<pybind11::object>(
      const_cast<AbstractMachine &>(*this).StateDict());
  for (auto value : state.attr("values")()) {
    pybind11::detail::array_proxy(value.ptr())->flags &=
        ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;
  }
  return state.release().ptr();
}

void AbstractMachine::StateDict(PyObject *other) {
  auto globals = pybind11::module::import("__main__").attr("__dict__");
  pybind11::dict locals;
  locals["copyto"] = pybind11::module::import("numpy").attr("copyto");
  locals["dst"] = pybind11::reinterpret_steal<pybind11::object>(StateDict());
  locals["src"] = pybind11::reinterpret_borrow<pybind11::object>(other);
  pybind11::exec(
      R"(
      for name, param in dst.items():
          if name not in src:
              raise ValueError("state is missing required field {!r}".format(name))
          value = src[name]
          if param is None:
              if value is not None:
                  raise ValueError("expected field {!r} to be None".format(name))
          else:
              copyto(param, value)
              assert (param == value).all()
      )",
      globals, locals);
}

namespace detail {
namespace {
/// Loads a Python object from a file \par filename using pickle.
//
/// \note This is a hacky solution which should be implemented in pure Python.
pybind11::object ReadFromFile(const std::string &filename) {
  pybind11::dict scope;
  scope["load"] = pybind11::module::import("pickle").attr("load");
  scope["filename"] = pybind11::cast(filename);
  scope["_return_value"] = pybind11::none();
  pybind11::exec(
      R"(
      with open(filename, "rb") as input:
          _return_value = load(input)
      )",
      pybind11::module::import("__main__").attr("__dict__"), scope);
  return static_cast<pybind11::object>(scope["_return_value"]);
}

/// Saves Python object \par obj to a file \par filename using pickle.
///
/// \note This is a hacky solution which should be implemented in pure Python.
void WriteToFile(const std::string &filename, pybind11::object obj) {
  pybind11::dict scope;
  scope["dump"] = pybind11::module::import("pickle").attr("dump");
  scope["filename"] = pybind11::cast(filename);
  scope["x"] = obj;
  pybind11::exec(
      R"(
      with open(filename, "wb") as output:
          dump(x, output)
      )",
      pybind11::module::import("__main__").attr("__dict__"), scope);
}
}  // namespace
}  // namespace detail

void AbstractMachine::Load(const std::string &filename) {
  auto const state = detail::ReadFromFile(filename);
  static_cast<AbstractMachine &>(*this).StateDict(state.ptr());
}

void AbstractMachine::Save(const std::string &filename) const {
  detail::WriteToFile(
      filename, pybind11::reinterpret_steal<pybind11::object>(StateDict()));
}

}  // namespace netket
