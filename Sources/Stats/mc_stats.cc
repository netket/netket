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

#include "Stats/mc_stats.hpp"

#include <mpi.h>
#include <nonstd/span.hpp>

#include "Utils/exceptions.hpp"

namespace netket {

namespace detail {
class MeanAndVarianceAccumulator {
  nonstd::span<double> const mu_;
  nonstd::span<double> const M2_;
  Index n_;

 public:
  MeanAndVarianceAccumulator(nonstd::span<double> mu, nonstd::span<double> var)
      : mu_{mu}, M2_{var}, n_{0} {
    CheckShape(__FUNCTION__, "var", mu_.size(), M2_.size());
    std::fill(mu_.begin(), mu_.end(), 0.0);
    std::fill(M2_.begin(), M2_.end(), 0.0);
  }

  void operator()(nonstd::span<const double> xs) {
    ++n_;
    const auto kernel = [this](double& mu, double& M2, const double x) {
      const auto delta = x - mu;
      mu += delta / n_;
      M2 += delta * (x - mu);
    };
    for (auto i = Index{0}; i < mu_.size(); ++i) {
      kernel(mu_[i], M2_[i], xs[i]);
    }
  }

  ~MeanAndVarianceAccumulator() {
    if (n_ == 0) {
      std::fill(mu_.begin(), mu_.end(),
                std::numeric_limits<double>::quiet_NaN());
      std::fill(M2_.begin(), M2_.end(),
                std::numeric_limits<double>::quiet_NaN());
    } else if (n_ == 1) {
      std::fill(M2_.begin(), M2_.end(),
                std::numeric_limits<double>::quiet_NaN());
    } else {
      // TODO: Are we __absolutely__ sure we want to divide by `n_` rather than
      // `n_ - 1`?
      const auto scale = 1.0 / static_cast<double>(n_);
      std::for_each(M2_.begin(), M2_.end(), [scale](double& x) { x *= scale; });
    }
  }
};
}  // namespace detail

/// \brief Computes in-chain means and variances.
std::pair<Eigen::VectorXcd, Eigen::VectorXd> StatisticsLocal(
    Eigen::Ref<const Eigen::VectorXcd> values, Index number_chains) {
  NETKET_CHECK(number_chains > 0, InvalidInputError,
               "invalid number of chains: " << number_chains
                                            << "; expected a positive integer");
  NETKET_CHECK(
      values.size() % number_chains == 0, InvalidInputError,
      "invalid number of chains: "
          << number_chains
          << "; `values.size()` must be a multiple of `number_chains`, but "
          << values.size() << " % " << number_chains << " = "
          << values.size() % number_chains);
  Eigen::VectorXcd mean(number_chains);
  // Buffer for variances of real and imaginary parts
  Eigen::VectorXd var(2 * number_chains);
  {
    detail::MeanAndVarianceAccumulator acc{
        nonstd::span<double>{reinterpret_cast<double*>(mean.data()),
                             2 * mean.size()},
        var};
    const auto* p = reinterpret_cast<const double*>(values.data());
    const auto n = values.size() / number_chains;
    for (auto i = Index{0}; i < n; ++i, p += 2 * number_chains) {
      acc({p, 2 * number_chains});
    }
    // Beware: destructor of acc does some work here!!
  }
  for (auto i = Index{0}; i < number_chains; ++i) {
    var(i) = var(2 * i) + var(2 * i + 1);
  }
  var.conservativeResize(number_chains);
  return std::make_pair(std::move(mean), std::move(var));
}

Stats Statistics(Eigen::Ref<const Eigen::VectorXcd> values,
                 Index local_number_chains) {
  NETKET_CHECK(values.size() >= local_number_chains, InvalidInputError,
               "not enough samples to compute statistics");
  constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();
  auto stats_local = StatisticsLocal(values, local_number_chains);
  // Number of samples in each Markov Chain
  const auto n = values.size() / local_number_chains;
  // Total number of Markov Chains we have:
  //   #processes x local_number_chains
  const auto m = []() {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return static_cast<Index>(size);
  }() * local_number_chains;

  // Calculates the mean over all Markov Chains
  const auto mean = [&stats_local, m]() {
    // Sum over Markov Chains on this MPI node
    Complex local_mean = stats_local.first.sum();
    // Sum over all MPI nodes
    Complex global_mean;
    MPI_Allreduce(&local_mean, &global_mean, 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
                  MPI_COMM_WORLD);
    // Average
    return global_mean / static_cast<double>(m);
  }();

  // (B / n, W)
  const auto var = [&stats_local, m, mean, NaN]() {
    double local_var[2] = {(stats_local.first.array() - mean).abs2().sum(),
                           stats_local.second.array().sum()};
    double global_var[2];
    MPI_Allreduce(&local_var, &global_var, 2, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    assert(m > 0);
    if (m == 1) {  // We can't estimate variance and error is there is only one
                   // chain.
      return std::make_pair(NaN, NaN);
    }
    // TODO: Are the coefficients here correct??
    return std::make_pair(global_var[0] / static_cast<double>(m - 1),
                          global_var[1] / static_cast<double>(m));
  }();

  if (!std::isnan(var.first) && !std::isnan(var.second)) {
    const auto t = var.first / var.second;
    auto correlation = 0.5 * (static_cast<double>(n) * t * t - 1.0);
    if (correlation < 0.0) correlation = NaN;
    const auto R =
        std::sqrt(static_cast<double>(n - 1) / static_cast<double>(n) +
                  var.first / var.second);
    return {mean, std::sqrt(var.first), var.second, correlation, R};
  }
  return Stats{mean, NaN, NaN, NaN, NaN};
}

}  // namespace netket
