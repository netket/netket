// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
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
#include "Utils/exceptions.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

struct Suggestion {
  nonstd::span<Index const> sites;
  nonstd::span<double const> values;
};

namespace detail {
class Flipper {
 public:
  template <class T>
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  template <class T>
  using RowMatrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  inline Flipper(std::pair<Index, Index> const shape,
                 std::vector<double> local_states);

  Index BatchSize() const noexcept { return state_.rows(); }
  Index SystemSize() const noexcept { return state_.cols(); }
  default_random_engine& Generator() noexcept { return engine_.Get(); }
  inline void Reset();
  inline void Next(nonstd::span<const bool> accept);
  inline Eigen::Ref<const RowMatrix<double>> Current() const noexcept;
  inline nonstd::span<Suggestion const> Read() const noexcept;
  inline void Read(Eigen::Ref<RowMatrix<double>> x) const noexcept;

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
  inline void RandomValues();

  /// \brief A vector of size `BatchSize()` which indices of sites at which we
  /// suggest changing quantum numbers
  Vector<Index> sites_;
  /// \brief A vector of size `BatchSize()` which contains new (proposed)
  /// quantum numbers.
  Vector<double> values_;
  /// \brief A matrix of size `BatchSize() x SystemSize()` with the
  /// current state of #BatchSize() different Markov chains.
  RowMatrix<double> state_;
  /// \brief Allowed values for quantum numbers
  std::vector<double> local_states_;

  std::vector<Suggestion> proposed_;
  DistributedRandomEngine engine_;
};
}  // namespace detail

class MetropolisLocalV2 {
  using InputType = detail::Flipper::RowMatrix<double>;
  using ForwardFn = std::function<
      auto(Eigen::Ref<const InputType>)->Eigen::Ref<const Eigen::VectorXcd>>;

  ForwardFn forward_;
  detail::Flipper flipper_;
  InputType proposed_X_;
  Eigen::ArrayXcd proposed_Y_;
  Eigen::ArrayXcd current_Y_;
  Eigen::ArrayXd randoms_;
  Eigen::Array<bool, Eigen::Dynamic, 1> accept_;

  inline MetropolisLocalV2(RbmSpinV2& machine, AbstractHilbert const& hilbert,
                           Index batch_size, std::true_type);

 public:
  MetropolisLocalV2(RbmSpinV2& machine, AbstractHilbert const& hilbert,
                    Index batch_size);

  Index BatchSize() const noexcept { return flipper_.BatchSize(); }
  Index SystemSize() const noexcept { return flipper_.SystemSize(); }

  std::pair<Eigen::Ref<const InputType>, Eigen::Ref<const Eigen::VectorXcd>>
  Read();
  void Next();
  void Reset();
};

struct StepsRange {
  StepsRange(std::tuple<Index, Index, Index> const& steps)
      : start_{std::get<0>(steps)},
        end_{std::get<1>(steps)},
        step_{std::get<2>(steps)} {
    CheckValid();
  }

  constexpr Index start() const noexcept { return start_; }
  constexpr Index end() const noexcept { return end_; }
  constexpr Index step() const noexcept { return step_; }
  constexpr Index size() const noexcept {
    return (end_ - start_ - 1) / step_ + 1;
  }

 private:
  void CheckValid() const;

  Index start_;
  Index end_;
  Index step_;
};  // namespace netket

std::tuple<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::VectorXcd>
ComputeSamples(MetropolisLocalV2& sampler, StepsRange const& steps);

}  // namespace netket

#endif  // SOURCES_SAMPLER_METROPOLIS_LOCAL_V2_HPP
