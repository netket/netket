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

#ifndef SOURCES_MACHINE_RBM_SPIN_V2_HPP
#define SOURCES_MACHINE_RBM_SPIN_V2_HPP

#include <cmath>
#include <memory>

#include <Eigen/Core>
#include <nonstd/optional.hpp>

#include "Hilbert/abstract_hilbert.hpp"
#include "Machine/abstract_machine.hpp"
#include "common_types.hpp"

namespace netket {

class RbmSpinV2 : public AbstractMachine {
 public:
  RbmSpinV2(std::shared_ptr<const AbstractHilbert> hilbert, Index nhidden,
            Index alpha, bool usea, bool useb, Index const batch_size);

  int Npar() const final {
    return W_.size() + (a_.has_value() ? a_->size() : 0) +
           (b_.has_value() ? b_->size() : 0);
  }
  int Nvisible() const final { return W_.rows(); }
  int Nhidden() const noexcept { return W_.cols(); }

  /// Returns current batch size.
  Index BatchSize() const noexcept;

  /// \brief Updates the batch size.
  ///
  /// There is no need to call this function explicitly -- batch size is changed
  /// automatically on calls to `LogVal` and `DerLog`.
  void BatchSize(Index batch_size);

  VectorType GetParameters() final;
  void SetParameters(Eigen::Ref<const Eigen::VectorXcd> pars) final;

  void LogVal(Eigen::Ref<const RowMatrix<double>> v,
              Eigen::Ref<Eigen::VectorXcd> out, const any & /*unused*/) final;

  void DerLog(Eigen::Ref<const RowMatrix<double>> v,
              Eigen::Ref<RowMatrix<Complex>> out, const any & /*unused*/) final;

  /// Simply calls `LogVal` with a batch size of 1.
  ///
  /// \note performance of this function is pretty bad. Please, use `LogVal`
  /// with batch sizes greater than 1 if at all possible.
  Complex LogValSingle(Eigen::Ref<const Eigen::VectorXd> v,
                       const any &cache) final {
    Complex data;
    auto out = Eigen::Map<Eigen::VectorXcd>(&data, 1);
    LogVal(v.transpose(), out, cache);
    return data;
  }

  /// Simply calls `DerLog` with a batch size of 1.
  ///
  /// \note performance of this function is pretty bad. Please, use `DerLog`
  /// with batch sizes greater than 1 if at all possible.
  Eigen::VectorXcd DerLogSingle(Eigen::Ref<const Eigen::VectorXd> v,
                                const any &cache) final {
    Eigen::VectorXcd out(Npar());
    DerLog(v.transpose(),
           Eigen::Map<RowMatrix<Complex>>{out.data(), 1, out.size()}, cache);
    return out;
  }

#if 0
  /// Simply calls `LogVal` twice.
  ///
  /// \note performance of this function is pretty bad. Please, restructure your
  /// code to avoid calling this function.
  Eigen::VectorXcd LogValDiff(
      Eigen::Ref<const Eigen::VectorXd> v,
      const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) final {
    RowMatrix<double> input(static_cast<Index>(tochange.size()), v.size());
    input = v.transpose().colwise().replicate(input.rows());
    for (auto i = Index{0}; i < input.rows(); ++i) {
      GetHilbert().UpdateConf(input.row(i), tochange[static_cast<size_t>(i)],
                              newconf[static_cast<size_t>(i)]);
    }
    auto x = AbstractMachine::LogVal(input, any{});
    x.array() -= LogValSingle(v, any{});
    return x;
  }

  /// Simply calls `LogVal` twice.
  ///
  /// \note performance of this function is pretty bad. Please, restructure your
  /// code to avoid calling this function.
  Complex LogValDiff(Eigen::Ref<const Eigen::VectorXd> v,
                     const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const any & /*unused*/) final {
    return LogValDiff(v, {tochange}, {newconf})(0);
  }
#endif

  // Look-up stuff
  any InitLookup(VisibleConstType) final { return {}; }
  void UpdateLookup(VisibleConstType, const std::vector<int> &,
                    const std::vector<double> &, any &) final {}

  void Save(const std::string &filename) const final;
  void Load(const std::string &filename) final;

  PyObject *StateDict() const final;
  PyObject *StateDict() final;
  void StateDict(PyObject *obj) final;

  bool IsHolomorphic() const noexcept final { return true; }

 private:
  /// Performs `out := log(cosh(out + b))`.
  void ApplyBiasAndActivation(Eigen::Ref<Eigen::VectorXcd> out) const;

  Eigen::MatrixXcd W_;             ///< weights
  nonstd::optional<VectorXcd> a_;  ///< visible units bias
  nonstd::optional<VectorXcd> b_;  ///< hidden units bias

  /// Caches
  RowMatrix<Complex> theta_;
};

}  // namespace netket

#endif  // SOURCES_MACHINE_RBM_SPIN_V2_HPP
