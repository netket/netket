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
#include "Sampler/metropolis_local_v2.hpp"
#include "Stats/binning.hpp"
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
  /// ???
  double correlation;
  /// Convergence estimator. The closer it is to 1, the better has the sampling
  /// converged.
  double R;
};

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
  Index num_chains;
};

#if 0
/**
 * Class storing the result data of a VMC run, i.e., a Markov chain of visible
 * configurations and corresponding log-derivatives of the wavefunction.
 */
class Result {
  MatrixXd samples_;
  nonstd::optional<MatrixXcd> log_derivs_;

 public:
  Result() = default;
  Result(Result &&) = default;
  Result &operator=(Result &&) = default;
  Result(const Result &) = delete;
  Result &operator=(const Result &) = delete;

  Result(MatrixXd samples, nonstd::optional<MatrixXcd> log_derivs)
      : samples_(std::move(samples)), log_derivs_(std::move(log_derivs)) {}

  /**
   * Returns the number of samples stored.
   */
  Index NSamples() const { return samples_.cols(); }

  /**
   * Returns a reference to the sample matrix V. Each column v_j is a visible
   * configuration vector.
   */
  const MatrixXd &SampleMatrix() const noexcept { return samples_; }

  /**
   * Returns a reference to a vector containing the j-th visible configuration
   * in the Markov chain.
   */
  Eigen::Ref<const VectorXd> Sample(Index j) const {
    assert(j >= 0 && j < NSamples());
    return samples_.col(j);
  }

  /**
   * Returns an optional reference to the matrix O of log-derivatives of the
   * wavefunction, which is non-empty if the log-derivs are present.
   * Define Δ_i(v_j) = ∂/∂x_i log Ψ(v_j). The values contained in O are
   * centered, i.e.,
   *      O_ij = Δ_i(v_j) - ⟨Δ_i⟩,
   * where ⟨Δ_i⟩ denotes the average over all samples.
   */
  const nonstd::optional<MatrixXcd> &LogDerivs() const noexcept {
    return log_derivs_;
  }
};
#endif

/**
 * Runs Monte Carlo sampling.
 *
 * @param sampler Sampler to use.
 * @param n_samples Minimal number of samples to generate. The actual number of
 *                  generated samples is \p n_samples rounded up to the closest
 *                  multiple of `sampler.BatchSize()`.
 * @param n_discard Number of #Sweep() s for warming up.
 * @param compute_gradients Whether to compute logarithmic derivatives of the
 *                          wavefunction.
 */
MCResult ComputeSamples(AbstractSampler &sampler, Index n_samples,
                        Index n_discard, bool compute_gradients);

/**
 * Computes the local values of the operator `op` in configurations `samples`.
 *
 * @param samples A matrix of MC samples as returned by #ComputeSamples(). Every
 *                row represents a single visible configuration.
 * @param values Logarithms of wave function values as returned by
 *               #ComputeSamples().
 * @param machine Machine representation of the wavefunction.
 * @param op Operator for which to compute the local values.
 * @param batch_size Batch size to use internally.
 *
 * @return local values of \p op
 */
Eigen::VectorXcd LocalValues(Eigen::Ref<const RowMatrix<double>> samples,
                             Eigen::Ref<const Eigen::VectorXcd> values,
                             AbstractMachine &machine,
                             const AbstractOperator &op, Index batch_size = 32);

/// Computes mean, variance, etc. given local estimators of an observable.
///
/// Since results from different markov chains are interleaved in `local_values`
/// is is impotant to correctly specify `local_number_chains`. One can obtain it
/// either from #AbstractSampler of #MCResult.
Stats Statistics(Eigen::Ref<const Eigen::VectorXcd> local_values,
                 Index local_number_chains);
#if 0
/**
 * Computes the local value of the operator `op` in configuration `v`
 * which is defined as O_loc(v) = ⟨v|op|Ψ⟩ / ⟨v|Ψ⟩.
 *
 * @param op Operator representing the observable.
 * @param psi Machine representation of the wavefunction.
 * @param v A many-body configuration.
 * @return The value of the local observable O_loc(v).
 */
Complex LocalValueLegacy(const AbstractOperator &op, AbstractMachine &psi,
                         Eigen::Ref<const VectorXd> v);
#endif

#if 0
/**
 * Computes the local values of the operator `op` in configurations `vs`.
 *
 * @param op Operator representing the observable.
 * @param psi Machine representation of the wavefunction.
 * @param vs A matrix of MC samples as returned by Result::SampleMatrix.
 * @param out A vector which will be filled with o_i = O_loc(v_i).
 *    The output vector will be resized as needed by this function.
 */
VectorXcd LocalValues(const AbstractOperator &op, AbstractMachine &psi,
                      Eigen::Ref<const MatrixXd> vs);
#endif

#if 0
/**
 * Computes the gradient of the local value with respect to the variational
 * parameters, ∇ O_loc, for an operator `op`.
 */
VectorXcd LocalValueDeriv(const AbstractOperator &op, AbstractMachine &psi,
                          Eigen::Ref<const VectorXd> v, VectorXcd &grad);

/**
 * Computes the expectation value of an operator based on VMC results.
 */
Stats Expectation(const Result &result, AbstractMachine &psi,
                  const AbstractOperator &op);

/**
 * Computes the expectation value of an operator based on VMC results.
 * The local value of the observable for each sampled configuration are stored
 * in the vector locvals and can be reused for further calculations.
 */
Stats Expectation(const Result &result, AbstractMachine &psi,
                  const AbstractOperator &op, VectorXcd &locvals);

/**
 * Computes the variance of an observable based on the given VMC data.
 */
Stats Variance(const Result &result, AbstractMachine &psi,
               const AbstractOperator &op);

/**
 * Computes variance of an observable based on the given VMC data.
 * This function reuses the precomputed expectation and local values of the
 * observable.
 */
Stats Variance(const Result &result, AbstractMachine &psi,
               const AbstractOperator &op, double expectation_value,
               const VectorXcd &locvals);
#endif

/**
 * Computes gradient of an observable with respect to the variational parameters
 * based on the given MC data.
 *
 * @param values Local values computed by #LocalValues().
 * @gradients gradients Logarithmic derivatives returned by #ComputeSamples().
 */
Eigen::VectorXcd Gradient(Eigen::Ref<const Eigen::VectorXcd> values,
                          Eigen::Ref<const RowMatrix<Complex>> gradients);

#if 0
/**
 * Computes the gradient of an observable with respect to the variational
 * parameters based on the given VMC data.
 */
VectorXcd Gradient(const Result &result, AbstractMachine &psi,
                   const AbstractOperator &op);

/**
 * Computes the gradient of an observable with respect to the variational
 * parameters based on the given VMC data. This function reuses the precomputed
 * expectation and local values of the observable.
 */
VectorXcd Gradient(const Result &result, AbstractMachine &psi,
                   const AbstractOperator &op, const VectorXcd &locvals);
#endif

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
