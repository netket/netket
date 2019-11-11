// Copyright 2018 The Simons Foundation, Inc. - All Rights
// Reserved.
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

#ifndef NETKET_EXACT_SAMPLER_HPP
#define NETKET_EXACT_SAMPLER_HPP

#include <Eigen/Core>

#include "Sampler/abstract_sampler.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"
#include "common_types.hpp"

namespace netket {

// Exact sampling using heat bath, mostly for testing purposes on small
// systems
class ExactSampler : public AbstractSampler {
  // number of visible units
  const int nv_;
  Index sample_size_;

  // states of visible units
  RowMatrix<double> current_v_;
  VectorXcd current_log_psi_;
  std::vector<int> state_index_;

  const HilbertIndex& hilbert_index_;

  const int dim_;

  using RNG = default_random_engine;
  std::discrete_distribution<int> dist_;

  std::vector<Complex> all_log_psi_vals_;
  std::vector<double> probability_mass_;

  ExactSampler(AbstractMachine& psi, Index sample_size, std::true_type)
      : AbstractSampler(psi),
        nv_(psi.GetHilbert().Size()),
        sample_size_(sample_size),
        current_v_(sample_size, nv_),
        current_log_psi_(sample_size),
        state_index_(sample_size),
        hilbert_index_(psi.GetHilbert().GetIndex()),
        dim_(psi.GetHilbert().GetIndex().NStates()) {
    NETKET_CHECK(psi.GetHilbert().IsDiscrete(), InvalidInputError,
                 "Exact sampler works only for discrete "
                 "Hilbert spaces");
    Reset(true);
  }

 public:
  ExactSampler(AbstractMachine& psi, Index sample_size)
      : ExactSampler(psi, detail::CheckNChains("ExactSampler", sample_size),
                     {}) {}

  void Reset(bool initrandom) override {
    double logmax = -std::numeric_limits<double>::infinity();

    all_log_psi_vals_.resize(dim_);
    probability_mass_.resize(dim_);

    for (int i = 0; i < dim_; ++i) {
      auto v = hilbert_index_.NumberToState(i);
      all_log_psi_vals_[i] = GetMachine().LogValSingle(v);
      logmax = std::max(logmax, std::real(all_log_psi_vals_[i]));
    }

    for (int i = 0; i < dim_; ++i) {
      probability_mass_[i] = NETKET_SAMPLER_APPLY_MACHINE_FUNC(
          std::exp(all_log_psi_vals_[i] - logmax));
    }

    dist_ = std::discrete_distribution<int>(probability_mass_.begin(),
                                            probability_mass_.end());

    if (initrandom) {
      Sweep();
    }
  }

  void Sweep() override {
    for (Index i = 0; i < sample_size_; ++i) {
      const auto idx = dist_(GetRandomEngine());
      state_index_[i] = idx;
      current_log_psi_(i) = all_log_psi_vals_[idx];
      current_v_.row(i) = hilbert_index_.NumberToState(idx);
    }
  }

  std::pair<Eigen::Ref<const RowMatrix<double>>,
            Eigen::Ref<const Eigen::VectorXcd>>
  CurrentState() const override {
    return {current_v_, current_log_psi_};
  }

  void SetVisible(Eigen::Ref<const RowMatrix<double>> v) override {
    CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()}, {sample_size_, nv_});
    current_v_ = v;
    for (Index i = 0; i < sample_size_; i++) {
      state_index_[i] = hilbert_index_.StateToNumber(current_v_.row(i));
    }
  }

  double Acceptance() const noexcept { return 1; }

  Index BatchSize() const noexcept override { return sample_size_; }

  Index NChains() const noexcept override { return sample_size_; }

  void SetMachineFunc(MachineFunction machine_func) override {
    AbstractSampler::SetMachineFunc(machine_func);
    Reset(true);
  }
};

}  // namespace netket

#endif
