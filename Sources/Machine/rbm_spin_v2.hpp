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

namespace netket {

// TODO: Remove me!
inline Complex SumLogCoshDumb(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> bias) {
  auto total = Complex{0.0, 0.0};
  for (auto i = Index{0}; i < input.size(); ++i) {
    total += std::log(std::cosh(input(i) + bias(i)));
  }
  return total;
}

class RbmSpinV2 : public AbstractMachine {
 public:
  template <class T>
  using RowMatrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  RbmSpinV2(std::shared_ptr<const AbstractHilbert> hilbert, Index nhidden,
            Index alpha, bool usea, bool useb, Index const batch_size);

  virtual int Npar() const override { return DoNpar(); }
  virtual int Nvisible() const override { return DoNvisible(); }
  int Nhidden() const noexcept { return DoNhidden(); }

  Index BatchSize() const noexcept;
  void BatchSize(Index batch_size);

  virtual VectorType GetParameters() override;
  virtual void SetParameters(Eigen::Ref<const Eigen::VectorXcd> pars) override;

  virtual void LogVal(Eigen::Ref<const RowMatrix<double>> v,
                      Eigen::Ref<Eigen::VectorXcd> out,
                      const any & /*unused*/) override;

  virtual void DerLog(Eigen::Ref<const RowMatrix<double>> v,
                      Eigen::Ref<RowMatrix<Complex>> out,
                      const any & /*unused*/) override;

  Eigen::Ref<Eigen::MatrixXcd> GetW() { return W_; }
  Eigen::Ref<Eigen::VectorXcd> GetA() { return a_.value(); }
  Eigen::Ref<Eigen::VectorXcd> GetB() { return b_.value(); }

  virtual Complex LogValSingle(Eigen::Ref<const Eigen::VectorXd> v,
                               const any &cache) override {
    Complex data;
    auto out = Eigen::Map<Eigen::VectorXcd>(&data, 1);
    LogVal(v.transpose(), out, cache);
    return data;
  }

  virtual Eigen::VectorXcd DerLogSingle(Eigen::Ref<const Eigen::VectorXd> v,
                                        const any &cache) override {
    Eigen::VectorXcd out(DoNpar());
    DerLog(v.transpose(),
           Eigen::Map<RowMatrix<Complex>>{out.data(), 1, out.size()}, cache);
    return out;
  }

  virtual Eigen::VectorXcd LogValDiff(
      Eigen::Ref<const Eigen::VectorXd> v,
      const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
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

  virtual Complex LogValDiff(Eigen::Ref<const Eigen::VectorXd> v,
                             const std::vector<int> &tochange,
                             const std::vector<double> &newconf,
                             const any & /*unused*/) override {
    return LogValDiff(v, {tochange}, {newconf})(0);
  }

  // Look-up stuff
  virtual any InitLookup(VisibleConstType) override { return {}; }
  virtual void UpdateLookup(VisibleConstType, const std::vector<int> &,
                            const std::vector<double> &, any &) override {}

  virtual void Save(const std::string &filename) const override { return; }
  virtual void Load(const std::string &filename) override { return; }

  virtual bool IsHolomorphic() const noexcept override { return true; }

 private:
  Index DoNhidden() const noexcept { return W_.cols(); }
  Index DoNvisible() const noexcept { return W_.rows(); }
  Index DoNpar() const noexcept {
    return W_.size() + (a_.has_value() ? a_->size() : 0) +
           (b_.has_value() ? b_->size() : 0);
  }

  void ApplyBiasAndActivation(Eigen::Ref<Eigen::VectorXcd> out) const;

  Eigen::MatrixXcd W_;             ///< weights
  nonstd::optional<VectorXcd> a_;  ///< visible units bias
  nonstd::optional<VectorXcd> b_;  ///< hidden units bias

  /// Caches
  RowMatrix<Complex> theta_;
};

}  // namespace netket

#endif  // SOURCES_MACHINE_RBM_SPIN_V2_HPP
