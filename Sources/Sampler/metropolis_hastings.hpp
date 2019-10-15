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

#ifndef SOURCES_SAMPLER_METROPOLISHASTINGS_HPP
#define SOURCES_SAMPLER_METROPOLISHASTINGS_HPP

#include <Eigen/Core>
#include <functional>
#include <limits>
#include <vector>

#include "Sampler/abstract_sampler.hpp"
#include "Utils/exceptions.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

class MetropolisHastings : public AbstractSampler {
 public:
  using TransitionKernel = std::function<void(
      Eigen::Ref<const RowMatrix<double>>, Eigen::Ref<RowMatrix<double>>,
      Eigen::Ref<Eigen::ArrayXd>)>;

 private:
  RowMatrix<double> current_X_;
  RowMatrix<double> proposed_X_;

  Eigen::ArrayXcd current_Y_;
  Eigen::ArrayXcd proposed_Y_;

  Eigen::ArrayXcd quotient_Y_;
  Eigen::ArrayXd probability_;

  Eigen::ArrayXd log_acceptance_correction_;

  Eigen::Array<bool, Eigen::Dynamic, 1> accept_;

  TransitionKernel transition_kernel_;

  Index n_chains_;
  Index sweep_size_;
  Index batch_size_;

  Index accepted_samples_;
  Index total_samples_;

 public:
  MetropolisHastings(AbstractMachine& ma, TransitionKernel tk, Index n_chains,
                     Index sweep_size, Index batch_size);

  Index BatchSize() const noexcept override;

  Index SweepSize() const noexcept;
  void SweepSize(Index sweep_size);

  Index NChains() const noexcept;

  /// Returns a batch of current visible states and corresponding log values.
  std::pair<Eigen::Ref<const RowMatrix<double>>,
            Eigen::Ref<const Eigen::VectorXcd>>
  CurrentState() const override;

  void SetVisible(Eigen::Ref<const RowMatrix<double>> x) override;

  void OneStep();

  void Sweep() override;

  /// Resets the sampler.
  void Reset(bool init_random) override;

  NETKET_SAMPLER_ACCEPTANCE_DEFAULT(accepted_samples_, total_samples_);

 private:
  void LogValBatched(Eigen::Ref<const RowMatrix<double>> v,
                     Eigen::Ref<AbstractMachine::VectorType> out,
                     const any& cache);
};

namespace detail {
inline Index CheckBatchSize(const char* func, const Index batch_size,
                            const Index n_chains) {
  if (batch_size <= 0) {
    std::ostringstream msg;
    msg << func << ": invalid batch size: " << batch_size
        << "; expected a positive number";
    throw InvalidInputError{msg.str()};
  }
  if (n_chains % batch_size != 0) {
    WarningMessage()
        << "The total number of chains is not an integer multiple of the "
           "batch_size, this can lead to suboptimal performance. \n";
  }
  if (n_chains < batch_size) {
    WarningMessage()
        << "The requested batch size is larger than the number of chains, the "
           "effective batch size used will be n_chains. \n";
  }
  return batch_size;
}
}  // namespace detail

}  // namespace netket

#endif
