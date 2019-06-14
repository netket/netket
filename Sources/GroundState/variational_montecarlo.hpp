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

#ifndef NETKET_VARIATIONALMONTECARLO_HPP
#define NETKET_VARIATIONALMONTECARLO_HPP

#include <complex>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <nonstd/optional.hpp>

#include "Machine/machine.hpp"
#include "Operator/abstract_operator.hpp"
#include "Optimizer/optimizer.hpp"
#include "Optimizer/stochastic_reconfiguration.hpp"
#include "Output/json_output_writer.hpp"
#include "Sampler/abstract_sampler.hpp"
#include "Sampler/vmc_sampling.hpp"
#include "Stats/stats.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "common_types.hpp"


namespace netket {

// Variational Monte Carlo schemes to learn the ground state
// Available methods:
// 1) Stochastic reconfiguration optimizer
//   both direct and sparse version
// 2) Gradient Descent optimizer
class VariationalMonteCarlo {
  const AbstractOperator &ham_;
  AbstractSampler &sampler_;
  AbstractMachine &psi_;


  int totalnodes_;
  int mynode_;

  AbstractOptimizer &opt_;
  SR sr_;
  bool dosr_;

  std::vector<const AbstractOperator *> obs_;
  std::vector<std::string> obsnames_;

  using StatsMap = std::unordered_map<std::string, vmc::Stats>;
  StatsMap observable_stats_;

  vmc::Result vmc_data_;
  Eigen::VectorXcd locvals_;
  Eigen::VectorXcd grad_;

  int nsamples_;
  int nsamples_node_;
  int ninitsamples_;
  int ndiscard_;

  int npar_;

  std::string target_;

 public:
  VariationalMonteCarlo(const AbstractOperator &hamiltonian,
                        AbstractSampler &sampler, AbstractOptimizer &optimizer,
                        int nsamples, int discarded_samples = -1,
                        int discarded_samples_on_init = 0,
                        const std::string &target = "energy",
                        const std::string &method = "Sr",
                        double diag_shift = 0.01, bool use_iterative = false,
                        bool use_cholesky = true)
      : ham_(hamiltonian),
        sampler_(sampler),
        psi_(sampler.GetMachine()),
        opt_(optimizer),
        target_(target) {
    Init(nsamples, discarded_samples, discarded_samples_on_init, method,
         diag_shift, use_iterative, use_cholesky);
  }

  void Init(int nsamples, int discarded_samples, int discarded_samples_on_init,
            const std::string &method, double diag_shift, bool use_iterative,
            bool use_cholesky) {
    npar_ = psi_.Npar();
    opt_.Init(npar_, psi_.IsHolomorphic());
    grad_.resize(npar_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    nsamples_ = nsamples;
    nsamples_node_ = int(std::ceil(double(nsamples_) / double(totalnodes_)));
    ninitsamples_ = discarded_samples_on_init;

    if (discarded_samples == -1) {
      ndiscard_ = 0.1 * nsamples_node_;
    } else {
      ndiscard_ = discarded_samples;
    }

    if (method == "Gd") {
      dosr_ = false;
      InfoMessage() << "Using a gradient-descent based method" << std::endl;
    } else {
      setSrParameters(diag_shift, use_iterative, use_cholesky);
    }

    if (target_ != "energy" && target_ != "variance") {
      InvalidInputError(
          "Target minimization should be either energy or variance\n");
    }

    InfoMessage() << "Variational Monte Carlo running on " << totalnodes_
                  << " processes" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void AddObservable(AbstractOperator &ob, const std::string &obname) {
    obs_.push_back(&ob);
    obsnames_.push_back(obname);
  }

  void InitSweeps() {
    sampler_.Reset();
    for (int i = 0; i < ninitsamples_; i++) {
      sampler_.Sweep();
    }
  }

  void Reset() {
    opt_.Reset();
    InitSweeps();
  }

  /**
   * Computes the expectation values of observables from the currently stored
   * samples.
   */
  void ComputeObservables() {
    for (std::size_t i = 0; i < obs_.size(); ++i) {
      auto ex = vmc::Expectation(vmc_data_, psi_, *obs_[i]);
      observable_stats_[obsnames_[i]] = ex;
    }
  }

  /**
   * Advances the simulation by performing `steps` VMC iterations.
   */
  void Advance(Index steps = 1) {
    assert(steps > 0);
    for (Index i = 0; i < steps; ++i) {
      vmc_data_ = vmc::ComputeSamples(sampler_, nsamples_node_, ndiscard_);

      const auto energy = vmc::Expectation(vmc_data_, psi_, ham_, locvals_);
      const auto variance =
          vmc::Variance(vmc_data_, psi_, ham_, energy.mean, locvals_);

      observable_stats_["Energy"] = energy;
      observable_stats_["EnergyVariance"] = variance;

      if (target_ == "energy") {
        grad_ = vmc::Gradient(vmc_data_, psi_, ham_, locvals_);
      } else if (target_ == "variance") {
        grad_ = vmc::GradientOfVariance(vmc_data_, psi_, ham_);
      } else {
        throw std::runtime_error("This should not happen.");
      }

      UpdateParameters();
    }
  }

  void Run(const std::string &output_prefix,
           nonstd::optional<Index> n_iter = nonstd::nullopt,
           Index step_size = 1, Index save_params_every = 50) {
    assert(n_iter > 0);
    assert(step_size > 0);
    assert(save_params_every > 0);

    nonstd::optional<JsonOutputWriter> writer;
    if (mynode_ == 0) {
      writer.emplace(output_prefix + ".log", output_prefix + ".wf",
                     save_params_every);
    }
    opt_.Reset();

    for (Index step = 0; !n_iter.has_value() || step < *n_iter;
         step += step_size) {
      Advance(step_size);
      ComputeObservables();

      // writer.has_value() iff the MPI rank is 0, so the output is only
      // written once
      if (writer.has_value()) {
        auto obs_data = json(observable_stats_);
        obs_data["Acceptance"] = sampler_.Acceptance();

        writer->WriteLog(step, obs_data);
        writer->WriteState(step, psi_);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  void UpdateParameters() {
    auto pars = psi_.GetParameters();

    Eigen::VectorXcd deltap(npar_);

    if (dosr_) {
      sr_.ComputeUpdate(vmc_data_.LogDerivs()->transpose(), grad_, deltap);
    } else {
      deltap = grad_;
    }
    opt_.Update(deltap, pars);

    SendToAll(pars);

    psi_.SetParameters(pars);

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void setSrParameters(double diag_shift = 0.01, bool use_iterative = false,
                       bool use_cholesky = true) {
    dosr_ = true;
    sr_.setParameters(diag_shift, use_iterative, use_cholesky,
                      psi_.IsHolomorphic());
  }

  AbstractMachine &GetMachine() { return psi_; }

  const StatsMap &GetObservableStats() const noexcept {
    return observable_stats_;
  }

  const vmc::Result &GetVmcData() const noexcept { return vmc_data_; }
};

}  // namespace netket

#endif
