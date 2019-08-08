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

#ifndef SOURCES_SAMPLER_METROPOLISHASTINGSPT_HPP
#define SOURCES_SAMPLER_METROPOLISHASTINGSPT_HPP

#include <Eigen/Core>
#include <functional>
#include <limits>
#include <vector>

#include "Sampler/abstract_sampler.hpp"
#include "Utils/exceptions.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Generic Metropolis-Hastings MCMC with parallel tempering
// a.k.a. replica exchange Monte Carlo
class MetropolisHastingsPt : public AbstractSampler {
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

  Eigen::ArrayXd beta_;
  Eigen::ArrayXd proposed_beta_;

  Eigen::ArrayXd log_acceptance_correction_;

  Eigen::Array<bool, Eigen::Dynamic, 1> accept_;

  TransitionKernel transition_kernel_;

  Index n_replicas_;
  Index sweep_size_;

  // Index holding the position of beta=1
  Index beta1_ind_;

 public:
  MetropolisHastingsPt(AbstractMachine& ma, TransitionKernel tk,
                       Index n_replicas, Index sweep_size);

  Index BatchSize() const noexcept override;

  Index SweepSize() const noexcept;
  void SweepSize(Index sweep_size);

  /// Returns a batch of current visible states and corresponding log values.
  std::pair<Eigen::Ref<const RowMatrix<double>>,
            Eigen::Ref<const Eigen::VectorXcd>>
  CurrentState() const override;

  void SetVisible(Eigen::Ref<const RowMatrix<double>> x) override;

  void OneStep();

  void ExchangeStep();

  void Sweep() override;

  /// Resets the sampler.
  void Reset(bool init_random) override;

  void ProposePairwiseSwap(Eigen::Ref<const Eigen::ArrayXd>,
                           Eigen::Ref<Eigen::ArrayXd>, int);
};

namespace detail {
inline Index CheckReplicasSize(const char* func, const Index n_replicas) {
  if (n_replicas < 2) {
    std::ostringstream msg;
    msg << func << ": invalid number of replicas : " << n_replicas
        << "; expected at least two replicas";
    throw InvalidInputError{msg.str()};
  }
  if (n_replicas % 2 != 0) {
    std::ostringstream msg;
    msg << func << ": invalid number of replicas : " << n_replicas
        << "; expected an even number of replicas";
    throw InvalidInputError{msg.str()};
  }

  return n_replicas;
}

}  // namespace detail

}  // namespace netket

#endif
