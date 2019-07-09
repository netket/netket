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

#include "Machine/rbm_spin_v2.hpp"

#include <pybind11/eigen.h>
#include <pybind11/eval.h>

#include "Utils/log_cosh.hpp"
#include "Utils/pybind_helpers.hpp"

namespace netket {

RbmSpinV2::RbmSpinV2(std::shared_ptr<const AbstractHilbert> hilbert,
                     Index nhidden, Index alpha, bool usea, bool useb,
                     Index const batch_size)
    : AbstractMachine{std::move(hilbert)},
      W_{},
      a_{nonstd::nullopt},
      b_{nonstd::nullopt},
      theta_{} {
  const auto nvisible = GetHilbert().Size();
  assert(nvisible >= 0 && "AbstractHilbert::Size is broken");
  if (nhidden < 0) {
    std::ostringstream msg;
    msg << "invalid number of hidden units: " << nhidden
        << "; expected a non-negative number";
    throw InvalidInputError{msg.str()};
  }
  if (alpha < 0) {
    std::ostringstream msg;
    msg << "invalid density of hidden units: " << alpha
        << "; expected a non-negative number";
    throw InvalidInputError{msg.str()};
  }
  if (nhidden > 0 && alpha > 0 && nhidden != alpha * nvisible) {
    std::ostringstream msg;
    msg << "number and density of hidden units are incompatible: " << nhidden
        << " != " << alpha << " * " << nvisible;
    throw InvalidInputError{msg.str()};
  }
  nhidden = std::max(nhidden, alpha * nvisible);

  W_.resize(nvisible, nhidden);
  if (usea) {
    a_.emplace(nvisible);
  }
  if (useb) {
    b_.emplace(nhidden);
  }

  theta_.resize(batch_size, nhidden);
}

Index RbmSpinV2::BatchSize() const noexcept { return theta_.rows(); }

void RbmSpinV2::BatchSize(Index batch_size) {
  if (batch_size <= 0) {
    std::ostringstream msg;
    msg << "invalid batch size: " << batch_size
        << "; expected a positive number";
    throw InvalidInputError{msg.str()};
  }
  if (batch_size != BatchSize()) {
    theta_.resize(batch_size, theta_.cols());
  }
}

Eigen::VectorXcd RbmSpinV2::GetParameters() {
  Eigen::VectorXcd parameters(Npar());
  Index i = 0;
  if (a_.has_value()) {
    parameters.segment(i, a_->size()) = *a_;
    i += a_->size();
  }
  if (b_.has_value()) {
    parameters.segment(i, b_->size()) = *b_;
    i += b_->size();
  }
  parameters.segment(i, W_.size()) =
      Eigen::Map<Eigen::VectorXcd>(W_.data(), W_.size());
  return parameters;
}

void RbmSpinV2::SetParameters(Eigen::Ref<const Eigen::VectorXcd> parameters) {
  if (parameters.size() != Npar()) {
    std::ostringstream msg;
    msg << "wrong shape: [" << parameters.size() << "]; expected [" << Npar()
        << "]";
    throw InvalidInputError{msg.str()};
  }
  Index i = 0;
  if (a_.has_value()) {
    *a_ = parameters.segment(i, a_->size());
    i += a_->size();
  }
  if (b_.has_value()) {
    *b_ = parameters.segment(i, b_->size());
    i += b_->size();
  }
  Eigen::Map<Eigen::VectorXcd>(W_.data(), W_.size()) =
      parameters.segment(i, W_.size());
}

void RbmSpinV2::LogVal(Eigen::Ref<const RowMatrix<double>> x,
                       Eigen::Ref<Eigen::VectorXcd> out, const any &) {
  if (x.cols() != Nvisible()) {
    std::ostringstream msg;
    msg << "input tensor has wrong shape: [" << x.rows() << ", " << x.cols()
        << "]; expected [?"
        << ", " << Nvisible() << "]";
    throw InvalidInputError{msg.str()};
  }
  if (out.size() != x.rows()) {
    std::ostringstream msg;
    msg << "output tensor wrong shape: [" << out.size() << "]; expected ["
        << x.rows() << "]";
    throw InvalidInputError{msg.str()};
  }
  BatchSize(x.rows());
  if (a_.has_value()) {
    out.noalias() = x * (*a_);
  } else {
    out.setZero();
  }
  theta_.noalias() = x * W_;
  ApplyBiasAndActivation(out);
}

void RbmSpinV2::DerLog(Eigen::Ref<const RowMatrix<double>> x,
                       Eigen::Ref<RowMatrix<Complex>> out,
                       const any & /*unused*/) {
  if (x.cols() != Nvisible()) {
    std::ostringstream msg;
    msg << "input tensor has wrong shape: [" << x.rows() << ", " << x.cols()
        << "]; expected [?"
        << ", " << Nvisible() << "]";
    throw InvalidInputError{msg.str()};
  }
  if (out.rows() != x.rows() || out.cols() != Npar()) {
    std::ostringstream msg;
    msg << "output tensor wrong shape: [" << out.rows() << ", " << out.cols()
        << "]; expected [" << x.rows() << ", " << Npar() << "]";
    throw InvalidInputError{msg.str()};
  }
  BatchSize(x.rows());

  auto i = Index{0};
  if (a_.has_value()) {
    out.block(0, i, BatchSize(), a_->size()) = x;
    i += a_->size();
  }

  Eigen::Map<RowMatrix<Complex>>{theta_.data(), theta_.rows(), theta_.cols()}
      .noalias() = x * W_;
  if (b_.has_value()) {
    theta_.array() = (theta_ + b_->transpose().colwise().replicate(BatchSize()))
                         .array()
                         .tanh();
    out.block(0, i, BatchSize(), b_->size()) = theta_;
    i += b_->size();
  } else {
    theta_.array() = theta_.array().tanh();
  }

  // TODO: Rewrite this using tensors
#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    Eigen::Map<Eigen::MatrixXcd>{&out(j, i), W_.rows(), W_.cols()}.noalias() =
        x.row(j).transpose() * theta_.row(j);
  }
}

void RbmSpinV2::ApplyBiasAndActivation(Eigen::Ref<Eigen::VectorXcd> out) const {
  if (b_.has_value()) {
#pragma omp parallel for schedule(static)
    for (auto j = Index{0}; j < BatchSize(); ++j) {
      out(j) += SumLogCosh(theta_.row(j), (*b_));  // total;
    }
  } else {
#pragma omp parallel for schedule(static)
    for (auto j = Index{0}; j < BatchSize(); ++j) {
      out(j) += SumLogCosh(theta_.row(j));
    }
  }
}

namespace detail {
namespace {
/// Saves Python object \par raw to a file \par filename using pickle.
///
/// \param raw reference to a Python object. It is borrowed, not stolen.
///
/// \note This is a hacky solution which should be implemented in pure Python.
void WriteToFile(const std::string &filename, PyObject *raw) {
  namespace py = pybind11;
  auto object = py::reinterpret_borrow<py::object>(raw);
  py::dict scope;
  scope["dump"] = pybind11::module::import("pickle").attr("dump");
  scope["filename"] = pybind11::cast(filename);
  scope["x"] = object;
  pybind11::exec(R"(
    with open(filename, "wb") as output:
        dump(x, output)
    )",
                 pybind11::module::import("__main__").attr("__dict__"), scope);
}

/// Loads a Python object from a file \par filename using pickle.
//
/// \return **new reference**.
///
/// \note This is a hacky solution which should be implemented in pure Python.
PyObject *ReadFromFile(const std::string &filename) {
  namespace py = pybind11;
  py::dict scope;
  scope["load"] = pybind11::module::import("pickle").attr("load");
  scope["filename"] = pybind11::cast(filename);
  scope["_return_value"] = pybind11::none();
  pybind11::exec(R"(
    with open(filename, "rb") as input:
        _return_value = load(input)
    )",
                 pybind11::module::import("__main__").attr("__dict__"), scope);
  return static_cast<py::object>(scope["_return_value"]).release().ptr();
}

/// Checks that \par src has the right shape to be written to \par dst.
template <class T1, class T2>
void CheckShape(char const *name, Eigen::EigenBase<T1> const &dst,
                Eigen::EigenBase<T2> const &src) {
  if (src.rows() != dst.rows() || src.cols() != dst.cols()) {
    std::ostringstream msg;
    msg << "field '" << name << "' has wrong shape: [" << src.rows() << ", "
        << src.cols() << "]; expected [" << dst.rows() << ", " << dst.cols()
        << "]";
    throw InvalidInputError{msg.str()};
  }
}

/// Loads some data from a Python object \par obj into an Eigen object
/// \par parameter. \par name should be the name of the parameter and is used
/// for error reporting.
template <class T>
void LoadImpl(char const *name, T &parameter, pybind11::object obj) {
  auto new_parameter = obj.cast<T>();
  CheckShape(name, parameter, new_parameter);
  parameter = std::move(new_parameter);
}

/// Loads some data from a Python object \par obj into an Eigen object
/// \par parameter. Very similar to #LoadImpl(char const*, T&, pybind11::object)
/// except that \par parameter may be `nullopt` in which case \par obj is
/// required to be `None`.
template <class T>
void LoadImpl(char const *name, nonstd::optional<T> &parameter,
              pybind11::object obj) {
  if (parameter.has_value()) {
    LoadImpl(name, *parameter, obj);
  } else if (!obj.is_none()) {
    std::ostringstream msg;
    msg << "expected field '" << name << "' to be None";
    throw InvalidInputError{msg.str()};
  }
}

/// Loads the `state[name]` into \par parameter.
template <class T>
void Load(char const *name, T &parameter, pybind11::dict state) {
  auto py_name = pybind11::str{name};
  if (!state.contains(py_name)) {
    std::ostringstream msg;
    msg << "state is missing required field '" << name << "'";
    throw InvalidInputError{msg.str()};
  }
  LoadImpl(name, parameter, state[py_name]);
}
}  // namespace
}  // namespace detail

void RbmSpinV2::Save(const std::string &filename) const {
  auto state = pybind11::reinterpret_steal<pybind11::dict>(StateDict());
  detail::WriteToFile(filename, state.ptr());
}

void RbmSpinV2::Load(const std::string &filename) {
  // NOTE: conversion to dict is very important, since it performs error
  // checking! So don't just change object's type to `pybind11::object`. The
  // code will compile, but as soon as the user passes some weird type, the
  // program will most likely crash.
  auto const state =
      pybind11::dict{pybind11::reinterpret_steal<pybind11::object>(
          detail::ReadFromFile(filename))};
  StateDict(state.ptr());
}

PyObject *RbmSpinV2::StateDict() const {
  namespace py = pybind11;
  py::dict state;
  state["a"] = a_.has_value() ? py::cast(*a_) : py::none();
  state["b"] = b_.has_value() ? py::cast(*b_) : py::none();
  state["w"] = py::cast(W_);
  return state.release().ptr();
}

void RbmSpinV2::StateDict(PyObject *obj) {
  namespace py = pybind11;
  auto state = py::dict{py::reinterpret_borrow<py::object>(obj)};
  detail::Load("a", a_, state);
  detail::Load("b", b_, state);
  detail::Load("w", W_, state);
}

}  // namespace netket
