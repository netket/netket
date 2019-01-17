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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <nonstd/optional.hpp>

#include "Machine/machine.hpp"
#include "Operator/abstract_operator.hpp"
#include "Optimizer/optimizer.hpp"
#include "Output/json_output_writer.hpp"
#include "Sampler/abstract_sampler.hpp"
#include "Stats/stats.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "common_types.hpp"
#include "matrix_replacement.hpp"

namespace netket {

namespace detail {
/**
 * Computes the average of the matrix or vector `t` of samples over all MPI
 * nodes.
 * @tparam T Eigen vector or matrix type
 * @param t Vector or matrix of samples (should have size n_samples or n_samples
 * x n_samples.
 * @param n_samples Number of samples per node.
 * @param n_nodes Number of MPI nodes.
 */
template <class T>
void AverageSamplesOverNodes(T &t, Index n_samples, Index n_nodes) {
  SumOnNodes(t);
  t /= double(n_samples) * n_nodes;
}
}  // namespace detail

/**
 * This class computes parameter updates for ground state search
 * using the stochastic reconfiguration method.
 */
class StochasticReconfiguration {
 public:
  enum class Solver {
    ConjugateGradient,
    CholeskyLLT,
    HouseholderQR,
  };

  explicit StochasticReconfiguration(
      double diag_shift = 0.0, bool rescale_shift = false,
      Solver linear_solver = Solver::ConjugateGradient)
      : diag_shift_(diag_shift),
        rescale_shift_(rescale_shift),
        solver_(linear_solver),
        n_nodes_(MpiSize()) {}

  Eigen::VectorXcd ComputeUpdate(const Eigen::MatrixXcd &Ok,
                                 const Eigen::VectorXcd &b) const {
    switch (solver_) {
      case Solver::ConjugateGradient:
      default:
        return Compute_ConjugateGradient(Ok, b);
        break;
      case Solver::CholeskyLLT:
        return Compute_CholeskyLLT(Ok, b);
        break;
      case Solver::HouseholderQR:
        return Compute_HouseholderQR(Ok, b);
        break;
    }
  }

 private:
  Eigen::VectorXcd Compute_ConjugateGradient(const Eigen::MatrixXcd &Ok,
                                             const Eigen::VectorXcd &b) const {
    double n_samples = Ok.rows();

    MatrixReplacement S;
    S.attachMatrix(Ok);
    S.setShift(diag_shift_);
    S.setScale(1. / n_samples * n_nodes_);

    using CGSolver =
        Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper,
                                 Eigen::IdentityPreconditioner>;
    CGSolver solver;
    solver.setTolerance(1.0e-3);  // TODO: Maybe make this configurable?

    solver.compute(S);
    Eigen::VectorXcd deltaP = solver.solve(b);

    if (rescale_shift_) {
      Complex nor = deltaP.dot(S * deltaP);
      deltaP /= std::sqrt(nor.real());
    }

    return deltaP;
  }

  Eigen::VectorXcd Compute_CholeskyLLT(const Eigen::MatrixXcd &Ok,
                                       const Eigen::VectorXcd &b) const {
    // Explicit construction of the S matrix
    Eigen::MatrixXcd S = Ok.adjoint() * Ok;
    const Index S_dim = S.rows();
    detail::AverageSamplesOverNodes(S, S_dim, n_nodes_);

    // Adding diagonal shift
    S += Eigen::MatrixXd::Identity(S_dim, S_dim) * diag_shift_;

    Eigen::LLT<Eigen::MatrixXcd> llt(S.rows());
    llt.compute(S);
    Eigen::VectorXcd deltaP = llt.solve(b);

    if (rescale_shift_) {
      Complex nor = (deltaP.dot(S * deltaP));
      deltaP /= std::sqrt(nor.real());
    }

    return deltaP;
  }

  Eigen::VectorXcd Compute_HouseholderQR(const Eigen::MatrixXcd &Ok,
                                         const Eigen::VectorXcd &b) const {
    // Explicit construction of the S matrix
    Eigen::MatrixXcd S = Ok.adjoint() * Ok;
    const Index S_dim = S.rows();
    detail::AverageSamplesOverNodes(S, S_dim, n_nodes_);

    // Adding diagonal shift
    S += Eigen::MatrixXd::Identity(S_dim, S_dim) * diag_shift_;

    Eigen::FullPivHouseholderQR<Eigen::MatrixXcd> qr(S.rows(), S.cols());
    qr.setThreshold(1.0e-6);  // TODO: Maybe make this configurable?
    qr.compute(S);
    Eigen::VectorXcd deltaP = qr.solve(b);

    if (rescale_shift_) {
      Complex nor = (deltaP.dot(S * deltaP));
      deltaP /= std::sqrt(nor.real());
    }

    return deltaP;
  }

  double diag_shift_;
  bool rescale_shift_;
  Solver solver_;
  Index n_nodes_;
};

// Variational Monte Carlo schemes to learn the ground state
// Available methods:
// 1) Stochastic reconfiguration optimizer
//   both direct and sparse version
// 2) Gradient Descent optimizer
class VariationalMonteCarlo {
  using GsType = Complex;
  using VectorT = Eigen::Matrix<typename AbstractMachine<GsType>::StateType,
                                Eigen::Dynamic, 1>;
  using MatrixT = Eigen::Matrix<typename AbstractMachine<GsType>::StateType,
                                Eigen::Dynamic, Eigen::Dynamic>;

  const AbstractOperator &ham_;
  AbstractSampler<AbstractMachine<GsType>> &sampler_;
  AbstractMachine<GsType> &psi_;

  std::vector<std::vector<int>> connectors_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  Eigen::VectorXcd elocs_;
  MatrixT Ok_;
  VectorT Okmean_;

  Eigen::MatrixXd vsamp_;

  Eigen::VectorXcd grad_;
  Eigen::VectorXcd gradprev_;

  int totalnodes_;

  int mynode_;
  AbstractOptimizer &opt_;

  std::vector<AbstractOperator *> obs_;

  std::vector<std::string> obsnames_;
  ObsManager obsmanager_;

  bool dosr_;
  StochasticReconfiguration sr_;

  int nsamples_;
  int nsamples_node_;
  int ninitsamples_;
  int ndiscardedsamples_;

  Complex elocmean_;
  double elocvar_;
  int npar_;

 public:
  class Iterator {
   public:
    // typedefs required for iterators
    using iterator_category = std::input_iterator_tag;
    using difference_type = Index;
    using value_type = Index;
    using pointer_type = Index *;
    using reference_type = Index &;

   private:
    VariationalMonteCarlo &vmc_;
    Index step_size_;
    nonstd::optional<Index> n_iter_;

    Index cur_iter_;

   public:
    Iterator(VariationalMonteCarlo &vmc, Index step_size,
             nonstd::optional<Index> n_iter)
        : vmc_(vmc),
          step_size_(step_size),
          n_iter_(std::move(n_iter)),
          cur_iter_(0) {}

    Index operator*() const { return cur_iter_; }

    Iterator &operator++() {
      vmc_.Advance(step_size_);
      cur_iter_ += step_size_;
      return *this;
    }

    // TODO(C++17): Replace with comparison to special Sentinel type, since
    // C++17 allows end() to return a different type from begin().
    bool operator!=(const Iterator &) {
      return !n_iter_.has_value() || cur_iter_ < n_iter_.value();
    }
    // pybind11::make_iterator requires operator==
    bool operator==(const Iterator &other) { return !(*this != other); }

    Iterator begin() const { return *this; }
    Iterator end() const { return *this; }
  };

  VariationalMonteCarlo(const AbstractOperator &hamiltonian,
                        AbstractSampler<AbstractMachine<GsType>> &sampler,
                        AbstractOptimizer &optimizer, int nsamples,
                        int discarded_samples = -1,
                        int discarded_samples_on_init = 0,
                        const std::string &method = "Sr",
                        double diag_shift = 0.01, bool rescale_shift = false,
                        bool use_iterative = false, bool use_cholesky = true)
      : ham_(hamiltonian),
        sampler_(sampler),
        psi_(sampler.GetMachine()),
        opt_(optimizer),
        elocvar_(0.),
        totalnodes_(MpiSize()),
        mynode_(MpiRank()) {
    Init(nsamples, discarded_samples, discarded_samples_on_init, method,
         diag_shift, rescale_shift, use_iterative, use_cholesky);
  }

  void Init(int nsamples, int discarded_samples, int discarded_samples_on_init,
            const std::string &method, double diagshift, bool rescale_shift,
            bool use_iterative, bool use_cholesky) {
    npar_ = psi_.Npar();

    opt_.Init(psi_.GetParameters());

    grad_.resize(npar_);
    Okmean_.resize(npar_);

    SetSrParameters();

    nsamples_ = nsamples;

    nsamples_node_ = int(std::ceil(double(nsamples_) / double(totalnodes_)));

    ninitsamples_ = discarded_samples_on_init;

    if (discarded_samples == -1) {
      ndiscardedsamples_ = 0.1 * nsamples_node_;
    } else {
      ndiscardedsamples_ = discarded_samples;
    }

    if (method == "Gd") {
      dosr_ = false;
    } else {
      dosr_ = true;
      SetSrParameters(diagshift, rescale_shift, use_iterative, use_cholesky);
    }

    if (dosr_) {
      InfoMessage() << "Using the Stochastic reconfiguration method"
                    << std::endl;

      if (use_iterative) {
        InfoMessage() << "With iterative solver" << std::endl;
      } else {
        if (use_cholesky) {
          InfoMessage() << "Using Cholesky decomposition" << std::endl;
        }
      }
    } else {
      InfoMessage() << "Using a gradient-descent based method" << std::endl;
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

  void Sample() {
    sampler_.Reset();

    for (int i = 0; i < ndiscardedsamples_; i++) {
      sampler_.Sweep();
    }

    vsamp_.resize(nsamples_node_, psi_.Nvisible());

    for (int i = 0; i < nsamples_node_; i++) {
      sampler_.Sweep();
      vsamp_.row(i) = sampler_.Visible();
    }
  }

  /**
   * Computes the expectation values of observables from the currently stored
   * samples.
   */
  void ComputeObservables() {
    const Index nsamp = vsamp_.rows();
    for (const auto &obname : obsnames_) {
      obsmanager_.Reset(obname);
    }
    for (Index i_samp = 0; i_samp < nsamp; ++i_samp) {
      for (std::size_t i_obs = 0; i_obs < obs_.size(); ++i_obs) {
        const auto &op = obs_[i_obs];
        const auto &name = obsnames_[i_obs];
        obsmanager_.Push(name, ObsLocValue(*op, vsamp_.row(i_samp)).real());
      }
    }
  }

  void Gradient() {
    obsmanager_.Reset("Energy");
    obsmanager_.Reset("EnergyVariance");

    const int nsamp = vsamp_.rows();
    elocs_.resize(nsamp);
    Ok_.resize(nsamp, psi_.Npar());

    for (int i = 0; i < nsamp; i++) {
      elocs_(i) = ObsLocValue(ham_, vsamp_.row(i));
      Ok_.row(i) = psi_.DerLog(vsamp_.row(i));
      obsmanager_.Push("Energy", elocs_(i).real());
    }

    elocmean_ = elocs_.mean();
    SumOnNodes(elocmean_);
    elocmean_ /= double(totalnodes_);

    Okmean_ = Ok_.colwise().mean();
    SumOnNodes(Okmean_);
    Okmean_ /= double(totalnodes_);

    Ok_ = Ok_.rowwise() - Okmean_.transpose();

    elocs_ -= elocmean_ * Eigen::VectorXd::Ones(nsamp);

    for (int i = 0; i < nsamp; i++) {
      obsmanager_.Push("EnergyVariance", std::norm(elocs_(i)));
    }

    grad_ = 2. * (Ok_.adjoint() * elocs_);
    detail::AverageSamplesOverNodes(grad_, nsamp, totalnodes_);
  }

  /**
   * Computes the value of the local estimator of the operator `ob` in
   * configuration `v` which is defined by O_loc(v) = ⟨v|ob|Ψ⟩ / ⟨v|Ψ⟩.
   *
   * @param ob Operator representing the observable.
   * @param v Many-body configuration
   * @return The value of the local observable O_loc(v).
   */
  Complex ObsLocValue(const AbstractOperator &ob, const Eigen::VectorXd &v) {
    ob.FindConn(v, mel_, connectors_, newconfs_);

    assert(connectors_.size() == mel_.size());

    auto logvaldiffs = (psi_.LogValDiff(v, connectors_, newconfs_));

    assert(mel_.size() == std::size_t(logvaldiffs.size()));

    Complex obval = 0;

    for (int i = 0; i < logvaldiffs.size(); i++) {
      obval += mel_[i] * std::exp(logvaldiffs(i));
    }

    return obval;
  }

  double ElocMean() { return elocmean_.real(); }

  double Elocvar() { return elocvar_; }

  void Advance(Index steps = 1) {
    assert(steps > 0);
    for (Index i = 0; i < steps; ++i) {
      Sample();
      Gradient();
      UpdateParameters();
    }
  }

  Iterator Iterate(const nonstd::optional<Index> &n_iter = nonstd::nullopt,
                   Index step_size = 1) {
    assert(!n_iter.has_value() || n_iter.value() > 0);
    assert(step_size > 0);

    opt_.Reset();
    InitSweeps();

    Advance(step_size);
    return Iterator(*this, step_size, n_iter);
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

    for (const auto step : Iterate(n_iter, step_size)) {
      // Note: This has to be called in all MPI processes, because converting
      // the ObsManager to JSON performs a MPI reduction.
      auto obs_data = json(obsmanager_);
      obs_data["Acceptance"] = sampler_.Acceptance();

      // writer.has_value() iff the MPI rank is 0, so the output is only
      // written once
      if (writer.has_value()) {
        writer->WriteLog(step, obs_data);
        writer->WriteState(step, psi_);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  void UpdateParameters() {
    auto pars = psi_.GetParameters();

    if (dosr_) {
      grad_ = sr_.ComputeUpdate(Ok_, 0.5 * grad_);
    }

    opt_.Update(grad_, pars);

    SendToAll(pars);

    psi_.SetParameters(pars);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void SetSrParameters(double diag_shift = 0.01, bool rescale_shift = false,
                       bool use_iterative = false, bool use_cholesky = true) {
    StochasticReconfiguration::Solver solver;
    if (use_iterative) {
      solver = StochasticReconfiguration::Solver::ConjugateGradient;
    } else if (use_cholesky) {
      solver = StochasticReconfiguration::Solver::CholeskyLLT;
    } else {
      solver = StochasticReconfiguration::Solver::HouseholderQR;
    }
    sr_ = StochasticReconfiguration{diag_shift, rescale_shift, solver};
  }

  AbstractMachine<Complex> &GetMachine() { return psi_; }
  const ObsManager &GetObsManager() const { return obsmanager_; }
};

}  // namespace netket

#endif
