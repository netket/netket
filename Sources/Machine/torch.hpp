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

#ifndef NETKET_SOURCES_MACHINE_TORCH_HPP
#define NETKET_SOURCES_MACHINE_TORCH_HPP

#include <torch/extension.h>
#include <torch/script.h>

#include "Machine/abstract_machine.hpp"

namespace netket {

class PyTorchMachine final : public AbstractMachine {
  /// \brief PyTorch module we are wrapping.
  ///
  /// \note This will become `torch::jit::script::Module` (without
  /// `std::shared_ptr`) when PyTorch v1.2 comes out.
  std::shared_ptr<torch::jit::script::Module> module_;
  /// \brief Cached `forward` method to avoid looking it up in a map every time.
  torch::jit::script::Method &forward_;
  /// \brief Total number of parameters in the machine.
  ///
  /// It is initialised during construction is never updated.
  Index n_par_;

  PyTorchMachine(std::shared_ptr<const AbstractHilbert> hilbert,
                 std::shared_ptr<torch::jit::script::Module> module);

 public:
  PyTorchMachine(std::shared_ptr<const AbstractHilbert> hilbert,
                 const std::string &filename);

  PyTorchMachine(const PyTorchMachine &) = delete;
  PyTorchMachine(PyTorchMachine &&) = delete;
  PyTorchMachine &operator=(const PyTorchMachine &) = delete;
  PyTorchMachine &operator=(PyTorchMachine &&) = delete;

  int Npar() const override { return static_cast<int>(n_par_); }
  bool IsHolomorphic() const noexcept override { return false; }

  void LogVal(Eigen::Ref<const RowMatrix<double>> v,
              Eigen::Ref<Eigen::VectorXcd> out,
              const any & /*unused*/) override;

  /// \brief Currently just throws a "not yet implemented" exception.
  void DerLog(Eigen::Ref<const RowMatrix<double>> v,
              Eigen::Ref<RowMatrix<Complex>> out,
              const any & /*unused*/) override;

  void JvpLog(Eigen::Ref<const RowMatrix<double>> v,
              Eigen::Ref<const VectorXcd> delta,
              Eigen::Ref<VectorXcd> out) override;

  Eigen::VectorXcd GetParameters() override;
  void SetParameters(Eigen::Ref<const Eigen::VectorXcd> pars) override;
  PyObject *StateDict() override;

  NETKET_MACHINE_DISABLE_LOOKUP
};

}  // namespace netket

#endif  // NETKET_SOURCES_MACHINE_TORCH_HPP
