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

  double sr_diag_shift_;
  bool sr_rescale_shift_;
  bool use_iterative_;

  int totalnodes_;
  int mynode_;

  AbstractOptimizer &opt_;

  std::vector<AbstractOperator *> obs_;
  std::vector<std::string> obsnames_;
  ObsManager obsmanager_;

  bool dosr_;

  bool use_cholesky_;

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
    nonstd::optional<Index> max_steps_;

    Index cur_iter_;

   public:
    Iterator(VariationalMonteCarlo &vmc, Index step_size,
             nonstd::optional<Index> max_steps)
        : vmc_(vmc),
          step_size_(step_size),
          max_steps_(std::move(max_steps)),
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
      return !max_steps_.has_value() || cur_iter_ < max_steps_.value();
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
        elocvar_(0.) {
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

    setSrParameters();

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

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
      setSrParameters(diagshift, rescale_shift, use_iterative, use_cholesky);
    }

    if (dosr_) {
      InfoMessage() << "Using the Stochastic reconfiguration method"
                    << std::endl;

      if (use_iterative_) {
        InfoMessage() << "With iterative solver" << std::endl;
      } else {
        if (use_cholesky_) {
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
        const auto& op = obs_[i_obs];
        const auto& name = obsnames_[i_obs];
        obsmanager_.Push(name, ComputeMean(*op, vsamp_.row(i_samp)));
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
      elocs_(i) = Eloc(vsamp_.row(i));
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

    // Summing the gradient over the nodes
    SumOnNodes(grad_);
    grad_ /= double(totalnodes_ * nsamp);
  }

  Complex Eloc(const Eigen::VectorXd &v) {
    ham_.FindConn(v, mel_, connectors_, newconfs_);

    assert(connectors_.size() == mel_.size());

    auto logvaldiffs = (psi_.LogValDiff(v, connectors_, newconfs_));

    assert(mel_.size() == std::size_t(logvaldiffs.size()));

    Complex eloc = 0;

    for (int i = 0; i < logvaldiffs.size(); i++) {
      eloc += mel_[i] * std::exp(logvaldiffs(i));
    }

    return eloc;
  }

  double ComputeMean(AbstractOperator &ob, const Eigen::VectorXd &v) {
    ob.FindConn(v, mel_, connectors_, newconfs_);

    assert(connectors_.size() == mel_.size());

    auto logvaldiffs = (psi_.LogValDiff(v, connectors_, newconfs_));

    assert(mel_.size() == std::size_t(logvaldiffs.size()));

    Complex obval = 0;

    for (int i = 0; i < logvaldiffs.size(); i++) {
      obval += mel_[i] * std::exp(logvaldiffs(i));
    }

    return obval.real();
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

  Iterator Iterate(const nonstd::optional<Index> &max_steps = nonstd::nullopt,
                   Index step_size = 1) {
    assert(!max_steps.has_value() || max_steps.value() > 0);
    assert(step_size > 0);

    opt_.Reset();
    InitSweeps();

    Advance(step_size);
    return Iterator(*this, step_size, max_steps);
  }

  void Run(const std::string &output_prefix,
           nonstd::optional<Index> max_steps = nonstd::nullopt,
           Index step_size = 1, Index save_params_every = 50) {
    assert(max_steps > 0);
    assert(step_size > 0);
    assert(save_params_every > 0);

    nonstd::optional<JsonOutputWriter> writer;
    if (mynode_ == 0) {
      writer.emplace(output_prefix + ".log", output_prefix + ".wf",
                     save_params_every);
    }

    for (const auto step : Iterate(max_steps, step_size)) {
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
      const int nsamp = vsamp_.rows();

      Eigen::VectorXcd b = Ok_.adjoint() * elocs_;
      SumOnNodes(b);
      b /= double(nsamp * totalnodes_);

      if (!use_iterative_) {
        // Explicit construction of the S matrix
        Eigen::MatrixXcd S = Ok_.adjoint() * Ok_;
        SumOnNodes(S);
        S /= double(nsamp * totalnodes_);

        // Adding diagonal shift
        S += Eigen::MatrixXd::Identity(pars.size(), pars.size()) *
             sr_diag_shift_;

        Eigen::VectorXcd deltaP;
        if (use_cholesky_ == false) {
          Eigen::FullPivHouseholderQR<Eigen::MatrixXcd> qr(S.rows(), S.cols());
          qr.setThreshold(1.0e-6);
          qr.compute(S);
          deltaP = qr.solve(b);
        } else {
          Eigen::LLT<Eigen::MatrixXcd> llt(S.rows());
          llt.compute(S);
          deltaP = llt.solve(b);
        }
        // Eigen::VectorXcd deltaP=S.jacobiSvd(ComputeThinU |
        // ComputeThinV).solve(b);

        assert(deltaP.size() == grad_.size());
        grad_ = deltaP;

        if (sr_rescale_shift_) {
          Complex nor = (deltaP.dot(S * deltaP));
          grad_ /= std::sqrt(nor.real());
        }

      } else {
        Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper,
                                 Eigen::IdentityPreconditioner>
            it_solver;
        // Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner>
        // it_solver;
        it_solver.setTolerance(1.0e-3);
        MatrixReplacement S;
        S.attachMatrix(Ok_);
        S.setShift(sr_diag_shift_);
        S.setScale(1. / double(nsamp * totalnodes_));

        it_solver.compute(S);
        auto deltaP = it_solver.solve(b);

        grad_ = deltaP;
        if (sr_rescale_shift_) {
          auto nor = deltaP.dot(S * deltaP);
          grad_ /= std::sqrt(nor.real());
        }

        // if(mynode_==0){
        //   std::cerr<<it_solver.iterations()<<"
        //   "<<it_solver.error()<<std::endl;
        // }
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }

    opt_.Update(grad_, pars);

    SendToAll(pars);

    psi_.SetParameters(pars);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void setSrParameters(double diagshift = 0.01, bool rescale_shift = false,
                       bool use_iterative = false, bool use_cholesky = true) {
    sr_diag_shift_ = diagshift;
    sr_rescale_shift_ = rescale_shift;
    use_iterative_ = use_iterative;
    dosr_ = true;
    use_cholesky_ = use_cholesky;
  }

  AbstractMachine<Complex> &GetMachine() { return psi_; }
  const ObsManager &GetObsManager() const { return obsmanager_; }
};

}  // namespace netket

#endif
