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

#ifndef NETKET_VMC_SAMPLING_HPP
#define NETKET_VMC_SAMPLING_HPP

#include "Machine/abstract_machine.hpp"
#include "Operator/abstract_operator.hpp"
#include "Sampler/abstract_sampler.hpp"
#include "common_types.hpp"

namespace netket {

/// Class storing the result data of a MC run.
struct MCResult {
  /// \brief Visible configurations visited during sampling.
  ///
  /// Every row represents a visible configuration.
  /// Samples from different Markov Chains are interleaved.
  RowMatrix<double> samples;
  /// \brief Logarithm of the wavefunction.
  ///
  /// `log_values(i) == LogVal(samples.row(i))`.
  Eigen::VectorXcd log_values;
  /// \brief Logarithmic derivatives with respect to variational parameters.
  nonstd::optional<RowMatrix<Complex>> der_logs;
  /// \brief Number of Markov Chains interleaved in #samples.
  Index n_chains;
};

/**
 * Runs Monte Carlo sampling.
 *
 * @param sampler Sampler to use.
 * @param n_samples Minimal number of samples to generate. The actual number of
 *                  generated samples is \p n_samples rounded up to the closest
 *                  multiple of `sampler.BatchSize()`.
 * @param n_discard Number of #Sweep() s for warming up.
 * @param der_logs  Whether to compute logarithmic derivatives of the
 *                  wavefunction. `nullopt` means don't compute the derivatives,
 *                  "normal" means compute the derivatives, and "centered" means
 *                  center them after computing.
 */
MCResult ComputeSamples(AbstractSampler &sampler, Index n_samples,
                        Index n_discard,
                        nonstd::optional<std::string> der_logs);

/**
 * Computes gradient of an observable with respect to the variational parameters
 * based on the given MC data.
 *
 * @param values Local values computed by #LocalValues().
 * @gradients gradients Logarithmic derivatives returned by #ComputeSamples().
 */
Eigen::VectorXcd Gradient(Eigen::Ref<const Eigen::VectorXcd> values,
                          Eigen::Ref<const RowMatrix<Complex>> gradients);

/**
 * Computes an approximation of the gradient of the variance of an operator.
 *
 * Specifically, the function returns ∇(σ²) = 2⟨∇O_loc (O_loc - ⟨O_loc⟩)⟩.
 * See Eq. (3) in Umrigar and Filippi, Phys. Rev. Lett. 94, 150201 (2005).
 */
VectorXcd GradientOfVariance(Eigen::Ref<const RowMatrix<double>> samples,
                             Eigen::Ref<const Eigen::VectorXcd> local_values,
                             AbstractMachine &psi, const AbstractOperator &op);

}  // namespace netket

#endif  // NETKET_VMC_SAMPLING_HPP
