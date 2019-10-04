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

class RbmSpin : public AbstractMachine {
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
              Eigen::Ref<Eigen::VectorXcd> out, const any &) final {
    SubBatch<Eigen::Ref<const RowMatrix<double>>, Eigen::Ref<Eigen::VectorXcd>>(
        std::bind(&RbmSpin::LogValImpl, this, std::placeholders::_1,
                  std::placeholders::_2),
        x, out);
  }

  void DerLog(Eigen::Ref<const RowMatrix<double>> x,
              Eigen::Ref<RowMatrix<Complex>> out,
              const any & /*unused*/) final {
    SubBatch<Eigen::Ref<const RowMatrix<double>>,
             Eigen::Ref<RowMatrix<Complex>>>(
        std::bind(&RbmSpin::DerLogImpl, this, std::placeholders::_1,
                  std::placeholders::_2),
        x, out);
  }

  void LogValImpl(Eigen::Ref<const RowMatrix<double>> x,
                  Eigen::Ref<Eigen::VectorXcd> out);

  void DerLogImpl(Eigen::Ref<const RowMatrix<double>> x,
                  Eigen::Ref<RowMatrix<Complex>> out);

  /// Simply calls `LogVal` with a batch size of 1.
  ///
  /// \note performance of this function is pretty bad. Please, use `LogVal`
  /// with batch sizes greater than 1 if at all possible.
  Complex LogValSingle(Eigen::Ref<const Eigen::VectorXd> v,
                       const any & /*unused*/) final {
    Complex data;
    auto out = Eigen::Map<Eigen::VectorXcd>(&data, 1);
    LogValImpl(v.transpose(), out);
    return data;
  }

  /// Simply calls `DerLog` with a batch size of 1.
  ///
  /// \note performance of this function is pretty bad. Please, use `DerLog`
  /// with batch sizes greater than 1 if at all possible.
  Eigen::VectorXcd DerLogSingle(Eigen::Ref<const Eigen::VectorXd> v,
                                const any & /*unused*/) final {
    Eigen::VectorXcd out(Npar());
    DerLogImpl(v.transpose(),
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
