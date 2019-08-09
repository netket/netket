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

#include "torch.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace netket {

namespace detail {

/// Runs a Depth-First Search on \p module.
///
/// \note \p function is copied on every recursive call so make sure you wrap
/// your function in a `std::reference_wrapper` if it is expensive to copy.
template <class T>
void ForEveryParameter(const torch::jit::script::Module& module, T function) {
  for (const auto& slot : module.get_parameters()) {
    function(slot);
  }
  for (const auto& child : module.get_modules()) {
    ForEveryParameter(*child, function);
  }
}

// NOTE: This function will probably fail if some of the parameters are not
// Tensors. Not sure if this is actually possible though
Index CountParameters(const torch::jit::script::Module& module) {
  auto n = Index{0};
  auto acc = [&n](const torch::jit::script::Slot& x) {
    n += x.value().toTensor().numel();
  };
  ForEveryParameter(module, std::cref(acc));
  return n;
}

torch::Tensor AsTensor(Eigen::Ref<const RowMatrix<double>> x) {
  return torch::from_blob(
      /*data=*/const_cast<double*>(x.data()),
      /*sizes=*/{x.rows(), x.cols()},
      /*strides=*/{x.cols(), 1}, torch::kFloat64);
}

std::shared_ptr<torch::jit::script::Module> LoadScriptModule(
    const std::string& filename) {
  auto m = torch::jit::load(filename);
  NETKET_CHECK(
      m != nullptr, RuntimeError,
      "Failed to load torch::jit::script::Module from '" << filename << "'");
  return m;
}

}  // namespace detail

PyTorchMachine::PyTorchMachine(std::shared_ptr<const AbstractHilbert> hilbert,
                               const std::string& filename)
    : PyTorchMachine{std::move(hilbert), detail::LoadScriptModule(filename)} {}

PyTorchMachine::PyTorchMachine(
    std::shared_ptr<const AbstractHilbert> hilbert,
    std::shared_ptr<torch::jit::script::Module> module)
    : AbstractMachine{std::move(hilbert)},
      module_{std::move(module)},
      forward_{module_->get_method("forward")},
      n_par_{detail::CountParameters(*module_)} {}

void PyTorchMachine::LogVal(Eigen::Ref<const RowMatrix<double>> v,
                            Eigen::Ref<Eigen::VectorXcd> out,
                            const any& /*unused*/) {
  CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()},
             {std::ignore, this->Nvisible()});
  CheckShape(__FUNCTION__, "out", out.size(), v.rows());
  auto r = forward_({detail::AsTensor(v)}).toTensor();
  const auto sizes = r.sizes();
  const auto strides = r.strides();
  const auto dtype = r.dtype();
  NETKET_CHECK((sizes == torch::IntArrayRef{v.rows(), 2}), RuntimeError,
               "user-defined network returned a tensor of wrong shape: "
                   << sizes << "; expected [" << v.rows() << ", 2]");
  NETKET_CHECK(r.scalar_type() == torch::kFloat64, RuntimeError,
               "user-defined network returned a tensor of wrong dtype: "
                   << dtype << "; expected Float64");
  NETKET_CHECK((strides == torch::IntArrayRef{2, 1}), RuntimeError,
               "it is assumed that by defaylt PyTorch creates row-major "
               "tensors; strides are"
                   << strides << "; expected [2, 1]");
  std::memcpy(out.data(), r.data_ptr(), 2 * out.size() * sizeof(double));
}

void PyTorchMachine::DerLog(Eigen::Ref<const RowMatrix<double>> /*unused*/,
                            Eigen::Ref<RowMatrix<Complex>> /*unused*/,
                            const any& /*unused*/) {
  NETKET_CHECK(false, RuntimeError, "not yet implemented");
}

void PyTorchMachine::JvpLog(Eigen::Ref<const RowMatrix<double>> v,
                            Eigen::Ref<const VectorXcd> delta,
                            Eigen::Ref<VectorXcd> out) {
  CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()},
             {std::ignore, this->Nvisible()});
  CheckShape(__FUNCTION__, "delta", delta.size(), v.rows());
  CheckShape(__FUNCTION__, "out", out.size(), Npar());

  const auto write_to_out = [this, &out](bool is_imag) {
    torch::NoGradGuard no_grad;
    // offset is 1 for imaginary part and 0 for real
    auto* ptr =
        const_cast<double*>(reinterpret_cast<const double*>(out.data())) +
        /*offset=*/is_imag;
    auto go = [&ptr](const torch::jit::script::Slot& slot) {
      auto src = slot.value().toTensor().grad().flatten();
      auto dst = torch::from_blob(ptr, src.sizes(), {2}, torch::kFloat64);
      dst.copy_(src);
      ptr += 2 * src.numel();
    };
    detail::ForEveryParameter(*module_, std::cref(go));
  };
  const auto zero_grad = [this]() {
    torch::NoGradGuard no_grad;
    py::print("<zero_grad>");
    auto go = [](const torch::jit::script::Slot& slot) {
      auto g = slot.value().toTensor().grad();
      if (g.defined()) g.zero_();
    };
    detail::ForEveryParameter(*module_, std::cref(go));
    py::print("</zero_grad>");
  };

  auto grad = torch::empty({delta.size(), 2}, torch::kFloat64);
  const auto get_delta = [&delta, &grad](bool is_imag) {
    py::print("<get_delta>");
    if (!is_imag) {
      grad.slice(/*dim=*/1, /*start=*/0, /*end=*/1)
          .squeeze()
          .copy_(torch::from_blob(
              const_cast<double*>(
                  reinterpret_cast<const double*>(delta.data()) +
                  /*offset=*/is_imag),
              {delta.size()}, {2}));
      grad.slice(/*dim=*/1, /*start=*/1, /*end=*/2).zero_();
    } else {
      grad.slice(/*dim=*/1, /*start=*/1, /*end=*/2)
          .squeeze()
          .copy_(torch::from_blob(
              const_cast<double*>(
                  reinterpret_cast<const double*>(delta.data()) +
                  /*offset=*/is_imag),
              {delta.size()}, {2}));
      grad.slice(/*dim=*/1, /*start=*/0, /*end=*/1).zero_();
    }
    py::print("</get_delta>");
    return grad;
  };

  auto y = forward_({detail::AsTensor(v)}).toTensor();
  y.backward(torch::randn({v.rows(), 2}, torch::kFloat64));
  assert(y.requires_grad());
  py::print("hello 1");
  zero_grad();
  py::print("hello 2");
  y.backward(get_delta(/*is_imag=*/false), /*keep_graph=*/true);
  py::print("hello 3");
  write_to_out(/*is_imag=*/false);
  py::print("hello 4");
  zero_grad();
  py::print("hello 5");
  y.backward(get_delta(/*is_imag=*/true), /*keep_graph=*/false);
  py::print("hello 6");
  write_to_out(/*is_imag=*/true);
  py::print("hello 7");
}

VectorXcd PyTorchMachine::GetParameters() {
  // First of all, we construct a list of all the parameters by running DFS on
  // the model
  std::vector<torch::Tensor> parameters;
  parameters.reserve(128);
  auto acc = [&parameters](const torch::jit::script::Slot& x) {
    parameters.emplace_back(x.value().toTensor().flatten());
  };
  torch::NoGradGuard no_grad;
  detail::ForEveryParameter(*module_, std::cref(acc));
  // The we ask PyTorch concatenate them
  VectorXcd out(Npar());
  auto out_tensor =
      torch::from_blob(out.data(), {out.size()}, {2}, torch::kFloat64);
  torch::cat_out(out_tensor, parameters);
  return out;
}

void PyTorchMachine::SetParameters(Eigen::Ref<const Eigen::VectorXcd> pars) {
  CheckShape(__FUNCTION__, "parameters", pars.size(), Npar());
  NETKET_CHECK((pars.imag().array() == 0.0).all(), InvalidInputError,
               "TorchMachine supports only real parameters; received a "
               "parameter with non-zero imaginary part");
  torch::NoGradGuard no_grad;
  auto* ptr = const_cast<double*>(reinterpret_cast<const double*>(pars.data()));
  auto go = [&ptr](const torch::jit::script::Slot& slot) {
    auto dst = slot.value().toTensor().flatten();
    auto src = torch::from_blob(ptr, dst.sizes(), {2}, torch::kFloat64);
    dst.copy_(src);
    ptr += 2 * dst.numel();
  };
  detail::ForEveryParameter(*module_, std::cref(go));
}

namespace detail {
namespace {
/// \brief Converts a Tensor into a NumPy array.
///
/// Kind of what `torch.Tensor.numpy` does in Python.
py::array AsNumPy(torch::Tensor const& x) {
  return AT_DISPATCH_ALL_TYPES(x.scalar_type(), "AsNumPy", [&]() -> py::array {
    const auto torch_strides = x.strides();
    // NumPy strides are in bytes rather than in counts of sizeof(scalar_t)
    std::vector<ssize_t> numpy_strides;
    numpy_strides.reserve(torch_strides.size());
    for (const auto s : torch_strides) {
      numpy_strides.push_back(sizeof(scalar_t) * s);
    }
    return py::array{pybind11::dtype::of<scalar_t>(), x.sizes(),
                     std::move(numpy_strides), x.data_ptr(), py::none()};
  });
}

/// \brief Builds a state dictionary for \p module.
///
/// See the implementation of `torch.nn.Module.state_dict` in PyTorch repo for /
/// more info.
py::object StateDict(const torch::jit::script::Module& module,
                     py::object destination = py::none(),
                     const std::string& prefix = "") {
  if (destination.is_none()) {
    destination = py::module::import("collections").attr("OrderedDict")();
  }
  for (const auto& slot : module.get_parameters()) {
    destination[py::str(prefix + slot.name())] =
        AsNumPy(slot.value().toTensor());
  }
  for (const auto& m : module.get_modules()) {
    StateDict(*m, destination, prefix + m->name() + '.');
  }
  return destination;
}
}  // namespace
}  // namespace detail

PyObject* PyTorchMachine::StateDict() {
  return detail::StateDict(*module_).release().ptr();
}

}  // namespace netket

// PYBIND11_MODULE(_C_netket_torch, m) { netket::AddPyTorchMachine(m); }
