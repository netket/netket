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
#include "Stats/binning.hpp"
#include "common_types.hpp"

namespace netket {
namespace vmc {

using Stats = Binning<double>::Stats;

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

/**
 * Computes a sequence of visible configurations based on Monte Carlo sampling
 * using `sampler`.
 *
 * @param sampler The sampler used to perform the MC sweeps.
 * @param nsamples The number of MC samples that are stored (one MC sweep is
 * performed between each sample)
 * @param ndiscard The number of sweeps to be discarded before starting to store
 * the samples.
 * @param compute_logderivs Whether to store the logarithmic derivatives of
 *    the wavefunction as part of the returned VMC result.
 * @return A Result object containing the MC samples and auxillary information.
 */
Result ComputeSamples(AbstractSampler &sampler, Index nsamples,
                      Index ndiscard = 0, bool compute_logderivs = true);

/**
 * Computes the local value of the operator `op` in configuration `v`
 * which is defined as O_loc(v) = ⟨v|op|Ψ⟩ / ⟨v|Ψ⟩.
 *
 * @param op Operator representing the observable.
 * @param psi Machine representation of the wavefunction.
 * @param v A many-body configuration.
 * @return The value of the local observable O_loc(v).
 */
Complex LocalValue(const AbstractOperator &op, AbstractMachine &psi,
                   Eigen::Ref<const VectorXd> v);

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

/**
 * Computes an approximation of the gradient of the variance of an operator.
 *
 * Specifically, the function returns ∇(σ²) = 2⟨∇O_loc (O_loc - ⟨O_loc⟩)⟩.
 * See Eq. (3) in Umrigar and Filippi, Phys. Rev. Lett. 94, 150201 (2005).
 */
VectorXcd GradientOfVariance(const Result &result, AbstractMachine &psi,
                             const AbstractOperator &op);

}  // namespace vmc
}  // namespace netket

#endif  // NETKET_VMC_SAMPLING_HPP
