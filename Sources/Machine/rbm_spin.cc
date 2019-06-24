// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
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

#include "rbm_spin.hpp"

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

#include "Utils/json_utils.hpp"
#include "Utils/messages.hpp"

namespace netket {

RbmSpin::RbmSpin(std::shared_ptr<const AbstractHilbert> hilbert, int nhidden,
                 int alpha, bool usea, bool useb)
    : AbstractMachine(hilbert),
      W_{},
      a_{},
      b_{},
      thetas_{},
      lnthetas_{},
      thetasnew_{},
      lnthetasnew_{} {
  const auto nvisible = hilbert->Size();
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

  thetas_.resize(nhidden);
  lnthetas_.resize(nhidden);
  thetasnew_.resize(nhidden);
  lnthetasnew_.resize(nhidden);
}

int RbmSpin::Nvisible() const { return W_.rows(); }

int RbmSpin::Npar() const {
  return W_.size() + (a_.has_value() ? a_->size() : 0) +
         (b_.has_value() ? b_->size() : 0);
}

void RbmSpin::InitRandomPars(int seed, double sigma) {
  VectorType par(Npar());

  netket::RandomGaussian(par, seed, sigma);

  SetParameters(par);
}

namespace detail {
namespace {
VectorXcd DenseFwd(AbstractMachine::MatrixType const &weights,
                   AbstractMachine::VisibleConstType x,
                   nonstd::optional<AbstractMachine::VectorType> const &bias) {
  return bias.has_value() ? (weights.transpose() * x + *bias).eval()
                          : (weights.transpose() * x).eval();
}
}  // namespace
}  // namespace detail

void RbmSpin::InitLookup(VisibleConstType v, LookupType &lt) {
  if (lt.VectorSize() == 0) {
    lt.AddVector(Nhidden());
  }
  if (lt.V(0).size() != Nhidden()) {
    lt.V(0).resize(Nhidden());
  }

  lt.V(0) = detail::DenseFwd(W_, v, b_);
}

void RbmSpin::UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                           const std::vector<double> &newconf, LookupType &lt) {
  if (tochange.size() != 0) {
    for (std::size_t s = 0; s < tochange.size(); s++) {
      const int sf = tochange[s];
      lt.V(0) += W_.row(sf) * (newconf[s] - v(sf));
    }
  }
}

RbmSpin::VectorType RbmSpin::DerLog(VisibleConstType v) {
  LookupType ltnew;
  InitLookup(v, ltnew);
  return DerLog(v, ltnew);
}

RbmSpin::VectorType RbmSpin::DerLog(VisibleConstType v, const LookupType &lt) {
  VectorType der(Npar());

  if (a_.has_value()) {
    der.head(Nvisible()) = v;
  }

  RbmSpin::tanh(lt.V(0), lnthetas_);

  if (b_.has_value()) {
    der.segment(a_.has_value() * Nvisible(), Nhidden()) = lnthetas_;
  }

  MatrixType wder = (v * lnthetas_.transpose());
  der.tail(W_.size()) = Eigen::Map<VectorType>(wder.data(), W_.size());

  return der;
}

RbmSpin::VectorType RbmSpin::GetParameters() {
  VectorType pars(Npar());

  if (a_.has_value()) {
    pars.head(Nvisible()) = *a_;
  }

  if (b_.has_value()) {
    pars.segment(a_.has_value() * Nvisible(), Nhidden()) = *b_;
  }

  pars.tail(W_.size()) = Eigen::Map<VectorType>(W_.data(), W_.size());

  return pars;
}

void RbmSpin::SetParameters(VectorConstRefType pars) {
  if (a_.has_value()) {
    a_ = pars.head(Nvisible());
  }

  if (b_.has_value()) {
    b_ = pars.segment(a_.has_value() * Nvisible(), Nhidden());
  }

  VectorType Wpars = pars.tail(W_.size());

  W_ = Eigen::Map<MatrixType>(Wpars.data(), Nvisible(), Nhidden());
}

// Value of the logarithm of the wave-function
Complex RbmSpin::LogVal(VisibleConstType v) {
  RbmSpin::lncosh(detail::DenseFwd(W_, v, b_), lnthetas_);

  return (a_.has_value() ? v.dot(*a_) : 0) + lnthetas_.sum();
}

// Value of the logarithm of the wave-function
// using pre-computed look-up tables for efficiency
Complex RbmSpin::LogVal(VisibleConstType v, const LookupType &lt) {
  RbmSpin::lncosh(lt.V(0), lnthetas_);

  return (a_.has_value() ? v.dot(*a_) : 0) + lnthetas_.sum();
}

// Difference between logarithms of values, when one or more visible variables
// are being flipped
RbmSpin::VectorType RbmSpin::LogValDiff(
    VisibleConstType v, const std::vector<std::vector<int>> &tochange,
    const std::vector<std::vector<double>> &newconf) {
  const std::size_t nconn = tochange.size();
  VectorType logvaldiffs = VectorType::Zero(nconn);

  thetas_ = detail::DenseFwd(W_, v, b_);
  RbmSpin::lncosh(thetas_, lnthetas_);

  Complex logtsum = lnthetas_.sum();

  if (a_.has_value()) {
    for (std::size_t k = 0; k < nconn; k++) {
      if (tochange[k].size() != 0) {
        thetasnew_ = thetas_;
        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];
          logvaldiffs(k) += (*a_)(sf) * (newconf[k][s] - v(sf));
          thetasnew_ += W_.row(sf) * (newconf[k][s] - v(sf));
        }
        RbmSpin::lncosh(thetasnew_, lnthetasnew_);
        logvaldiffs(k) += lnthetasnew_.sum() - logtsum;
      }
    }
  } else {
    for (std::size_t k = 0; k < nconn; k++) {
      if (tochange[k].size() != 0) {
        thetasnew_ = thetas_;
        for (std::size_t s = 0; s < tochange[k].size(); s++) {
          const int sf = tochange[k][s];
          thetasnew_ += W_.row(sf) * (newconf[k][s] - v(sf));
        }
        RbmSpin::lncosh(thetasnew_, lnthetasnew_);
        logvaldiffs(k) += lnthetasnew_.sum() - logtsum;
      }
    }
  }
  return logvaldiffs;
}

// Difference between logarithms of values, when one or more visible variables
// are being flipped Version using pre-computed look-up tables for efficiency
// on a small number of spin flips
Complex RbmSpin::LogValDiff(VisibleConstType v,
                            const std::vector<int> &tochange,
                            const std::vector<double> &newconf,
                            const LookupType &lt) {
  Complex logvaldiff = 0.;
  if (tochange.size() != 0) {
    RbmSpin::lncosh(lt.V(0), lnthetas_);
    thetasnew_ = lt.V(0);
    if (a_.has_value()) {
      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        logvaldiff += (*a_)(sf) * (newconf[s] - v(sf));
        thetasnew_ += W_.row(sf) * (newconf[s] - v(sf));
      }
    } else {
      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        thetasnew_ += W_.row(sf) * (newconf[s] - v(sf));
      }
    }
    RbmSpin::lncosh(thetasnew_, lnthetasnew_);
    logvaldiff += (lnthetasnew_.sum() - lnthetas_.sum());
  }
  return logvaldiff;
}

namespace detail {
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
}  // namespace detail

void RbmSpin::Save(const std::string &filename) const {
  auto state = pybind11::reinterpret_steal<pybind11::dict>(StateDict());
  detail::WriteToFile(filename, state.ptr());
}

void RbmSpin::Load(const std::string &filename) {
  // NOTE: conversion to dict is very important, since it performs error
  // checking! So don't just change object's type to `pybind11::object`. The
  // code will compile, but as soon as the user passes some weird type, the
  // program will most likely crash.
  auto const state =
      pybind11::dict{pybind11::reinterpret_steal<pybind11::object>(
          detail::ReadFromFile(filename))};
  StateDict(state.ptr());
}

bool RbmSpin::IsHolomorphic() const noexcept { return true; }

PyObject *RbmSpin::StateDict() const {
  namespace py = pybind11;
  py::dict state;
  state["a"] = a_.has_value() ? py::cast(*a_) : py::none();
  state["b"] = b_.has_value() ? py::cast(*b_) : py::none();
  state["w"] = py::cast(W_);
  return state.release().ptr();
}

namespace detail {
namespace {
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

void RbmSpin::StateDict(PyObject *obj) {
  namespace py = pybind11;
  auto state = py::dict{py::reinterpret_borrow<py::object>(obj)};
  detail::Load("a", a_, state);
  detail::Load("b", b_, state);
  detail::Load("w", W_, state);
}

}  // namespace netket
