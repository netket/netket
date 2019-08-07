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

#ifndef SOURCES_SAMPLER_METROPOLIS_LOCAL_V2_HPP
#define SOURCES_SAMPLER_METROPOLIS_LOCAL_V2_HPP

#include <functional>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <nonstd/span.hpp>

#include "Machine/rbm_spin_v2.hpp"
#include "Sampler/abstract_sampler.hpp"
#include "Utils/exceptions.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

struct ConfDiff {
  nonstd::span<Index const> sites;
  nonstd::span<double const> values;
};

namespace detail {
/// Suggests which spins to try flipping next.
class Flipper {
 public:
  template <class T>
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  inline Flipper(std::pair<Index, Index> const shape,
                 std::vector<double> local_states,
                 default_random_engine& engine);

  Index BatchSize() const noexcept { return state_.rows(); }
  Index Nvisible() const noexcept { return state_.cols(); }

  /// \brief Resets the flipper.
  ///
  /// Randomizes internal state.
  inline void Reset();

  /// \brief Makes a move.
  ///
  /// \param accept has length #BatchSize() and describes which flips were
  /// accepted and which weren't (`accept[i]==true` means that the `i`th flip
  /// was accepted).
  inline void Update(nonstd::span<const bool> accept);

  /// \brief Returns the current visible state.
  ///
  /// Each row of the matrix describes one visible configuration. There are
  /// #BatchSize() rows in the returned matrix.
  inline const RowMatrix<double>& Visible() const noexcept;

  /// \brief Returns the current visible state.
  ///
  /// Each row of the matrix describes one visible configuration. There are
  /// #BatchSize() rows in the returned matrix.
  inline RowMatrix<double>& Visible() noexcept;

  /// \brief Returns next spins to try flipping.
  ///
  /// \warning Don't call this function more than once per call to #Update()!
  inline nonstd::span<ConfDiff const> Propose();

  /// \brief Similar to #Read() except that the result is written into \p x.
  ///
  /// \warning Don't call this function more than once per call to #Update()!
  inline void Propose(Eigen::Ref<RowMatrix<double>> x);

  inline nonstd::span<const double> LocalStates() const noexcept;

 private:
  /// \brief Randomizes the state.
  ///
  /// New state is sampled uniformly from #local_states_
  inline void RandomState();

  /// \brief Randomizes the indices.
  ///
  /// New indices are sampled uniformly.
  inline void RandomSites();

  /// \brief Randomizes the values.
  ///
  /// New values are chosen uniformly from all possible quantum numbers except
  /// the current one (to minimize staying in the same configuration).
  inline void RandomNewValues();

  /// \brief A vector of size `BatchSize()` which indices of sites at which we
  /// suggest changing quantum numbers
  Vector<Index> sites_;
  /// \brief A vector of size `BatchSize()` which contains new (proposed)
  /// quantum numbers.
  Vector<double> new_values_;
  /// \brief A matrix of size `BatchSize() x Nvisible()` with the
  /// current state of #BatchSize() different Markov chains.
  RowMatrix<double> state_;
  /// \brief Allowed values for quantum numbers
  std::vector<double> local_states_;

  std::vector<ConfDiff> proposed_;
  default_random_engine& engine_;
};
}  // namespace detail

class MetropolisLocalV2 : public AbstractSampler {
  detail::Flipper flipper_;
  RowMatrix<double> proposed_X_;
  Eigen::ArrayXcd proposed_Y_;
  Eigen::ArrayXcd current_Y_;
  Eigen::ArrayXcd quotient_Y_;
  Eigen::ArrayXd probability_;
  Eigen::Array<bool, Eigen::Dynamic, 1> accept_;
  Index sweep_size_;

  std::size_t accepted_samples_ = 0;
  std::size_t total_samples_ = 0;

  inline MetropolisLocalV2(AbstractMachine& machine, Index batch_size,
                           Index sweep_size, std::true_type);

  /// Makes a step.
  void Next();

 public:
  MetropolisLocalV2(AbstractMachine& machine, Index batch_size,
                    Index sweep_size);

  Index BatchSize() const noexcept override { return flipper_.BatchSize(); }
  Index Nvisible() const noexcept { return flipper_.Nvisible(); }

  Index SweepSize() const noexcept { return sweep_size_; }
  void SweepSize(Index sweep_size);

  /// Returns a batch of current visible states and corresponding log values.
  std::pair<Eigen::Ref<const RowMatrix<double>>,
            Eigen::Ref<const Eigen::VectorXcd>>
  CurrentState() const override;

  void SetVisible(Eigen::Ref<const RowMatrix<double>> x) override;

  void Sweep() override;

  /// Resets the sampler.
  void Reset(bool init_random) override;

  NETKET_SAMPLER_ACCEPTANCE_DEFAULT(accepted_samples_, total_samples_);
};

}  // namespace netket

#endif  // SOURCES_SAMPLER_METROPOLIS_LOCAL_V2_HPP
