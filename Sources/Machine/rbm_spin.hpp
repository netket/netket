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

#ifndef SOURCES_MACHINE_RBM_SPIN_HPP
#define SOURCES_MACHINE_RBM_SPIN_HPP

#include <cmath>
#include <memory>

#include <Eigen/Core>
#include <nonstd/optional.hpp>

#include "Hilbert/abstract_hilbert.hpp"
#include "Machine/abstract_machine.hpp"
#include "common_types.hpp"

namespace netket {

/// Restricted Boltzmann machine with spin 1/2 hidden units.
class RbmSpin : public AbstractMachine {
/* Variables are:
 * hilbert - the variable containing information about physical system
 * nhidden - number of neurons in hidden layer defined by larger of the values
 *           (alpha * number of neuron from visible layer) or nhidden
 * alpha - density defined as (number of neurons in hidden layer / number of
 *         neurons in visible layer)
 * usea - if true use biases in visible layer a*output
 * useb - if true use biases in hidden layer b*output
 *
 * Defining variational parameters: weights and biases (a_ for
 * visible layer, b_ for hidden layer)
 */
 public:
  RbmSpin(std::shared_ptr<const AbstractHilbert> hilbert, Index nhidden,
          Index alpha, bool usea, bool useb);

  int Npar() const final {
    return W_.size() + (a_.has_value() ? a_->size() : 0) +
           (b_.has_value() ? b_->size() : 0);
  }
  int Nvisible() const final { return W_.rows(); }
  int Nhidden() const noexcept { return W_.cols(); }

  VectorType GetParameters() final;
  void SetParameters(Eigen::Ref<const Eigen::VectorXcd> pars) final;

  void LogVal(Eigen::Ref<const RowMatrix<double>> x,
              Eigen::Ref<Eigen::VectorXcd> out, const any &lt = any{}) final;

  void DerLog(Eigen::Ref<const RowMatrix<double>> x,
              Eigen::Ref<RowMatrix<Complex>> out, const any &lt = any{}) final;

  /// Simply calls `LogVal` with a batch size of 1.
  ///
  /// \note performance of this function is pretty bad. Please, use `LogVal`
  /// with batch sizes greater than 1 if at all possible.
  Complex LogValSingle(Eigen::Ref<const Eigen::VectorXd> v,
                       const any &lt) final {
    Complex data;
    auto out = Eigen::Map<Eigen::VectorXcd>(&data, 1);
    LogVal(v.transpose(), out, lt);
    return data;
  }

  /// Simply calls `DerLog` with a batch size of 1.
  ///
  /// \note performance of this function is pretty bad. Please, use `DerLog`
  /// with batch sizes greater than 1 if at all possible.
  Eigen::VectorXcd DerLogSingle(Eigen::Ref<const Eigen::VectorXd> v,
                                const any & /*unused*/) final {
    Eigen::VectorXcd out(Npar());
    DerLog(v.transpose(),
           Eigen::Map<RowMatrix<Complex>>{out.data(), 1, out.size()});
    return out;
  }

  PyObject *StateDict() final;

  bool IsHolomorphic() const noexcept final { return true; }

 private:
  /// Performs `out := log(cosh(out + b))`.
  void ApplyBiasAndActivation(Eigen::Ref<Eigen::VectorXcd> out) const;

  Eigen::MatrixXcd W_;             ///< weights
  nonstd::optional<VectorXcd> a_;  ///< visible units bias
  nonstd::optional<VectorXcd> b_;  ///< hidden units bias

  /// Caches
  RowMatrix<Complex> theta_;

  /// Returns current batch size.
  Index BatchSize() const noexcept;

  /// \brief Updates the batch size.
  void BatchSize(Index batch_size);
};

}  // namespace netket

#endif  // SOURCES_MACHINE_RBM_SPIN_V2_HPP
