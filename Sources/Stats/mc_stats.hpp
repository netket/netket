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

#ifndef NETKET_STATS_MC_STATS_HPP
#define NETKET_STATS_MC_STATS_HPP

#include "common_types.hpp"

namespace netket {

struct Stats {
  /// Mean value of the observable over all Markov Chains.
  Complex mean;
  /// Standard deviation of the mean values of all chains.
  /// TODO: Make this split chains into halfs.
  double error_of_mean;
  /// Average of in-chain variances of the observable over all Markov Chains.
  double variance;
  /// Autocorrelation time
  double correlation;
  /// Convergence estimator. The closer it is to 1, the better has the sampling
  /// converged.
  double R;
};

/// Computes mean, variance, etc. given local estimators of an observable.
///
/// Since results from different markov chains are interleaved in `local_values`
/// is is impotant to correctly specify `local_number_chains`. One can obtain it
/// either from #AbstractSampler of #MCResult.
Stats Statistics(Eigen::Ref<const Eigen::VectorXcd> local_values,
                 Index local_number_chains);

Eigen::VectorXcd product_sv(Eigen::Ref<const Eigen::VectorXcd> s_values,
                            Eigen::Ref<const RowMatrix<Complex>> v_values);

void SubtractMean(Eigen::Ref<RowMatrix<Complex>> v_values);

}  // namespace netket

#endif  // NETKET_STATS_MC_STATS_HPP
