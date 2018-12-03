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

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "Machine/machine.hpp"
#include "Operator/operator.hpp"
#include "Optimizer/optimizer.hpp"
#include "Output/json_output_writer.hpp"
#include "Sampler/sampler.hpp"
#include "Stats/stats.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "matrix_replacement.hpp"

namespace netket {

// Variational Monte Carlo schemes to learn the ground state
// Available methods:
// 1) Stochastic reconfiguration optimizer
//   both direct and sparse version
// 2) Gradient Descent optimizer
class VariationalMonteCarlo {
  using GsType = std::complex<double>;

  using VectorT = Eigen::Matrix<typename AbstractMachine<GsType>::StateType,
                                Eigen::Dynamic, 1>;
  using MatrixT = Eigen::Matrix<typename AbstractMachine<GsType>::StateType,
                                Eigen::Dynamic, Eigen::Dynamic>;

  const Operator ham_;
  Sampler<Machine<GsType>> sampler_;
  Machine<GsType> psi_;

  std::vector<std::vector<int>> connectors_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<std::complex<double>> mel_;

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

  // This optional will contain a value iff the MPI rank is 0.
  nonstd::optional<JsonOutputWriter> output_;

  Optimizer opt_;

  std::vector<Operator> obs_;
  std::vector<std::string> obsnames_;
  ObsManager obsmanager_;

  bool dosr_;

  bool use_cholesky_;

  int nsamples_;
  int nsamples_node_;
  int ninitsamples_;
  int ndiscardedsamples_;
  int niter_opt_;

  std::complex<double> elocmean_;
  double elocvar_;
  int npar_;

 public:
  VariationalMonteCarlo(Operator ham, Sampler<Machine<GsType>> sampler,
                        Optimizer opt, int nsamples, int niter_opt,
                        std::string output_file, int discarded_samples = -1,
                        int discarded_samples_on_init = 0,
                        std::string method = "Sr", double diagshift = 0.01,
                        bool rescale_shift = false, bool use_iterative = false,
                        bool use_cholesky = true, int save_every = 50)
      : ham_(ham),
        sampler_(sampler),
        psi_(sampler_.GetMachine()),
        opt_(opt),
        elocvar_(0.) {
    Init(nsamples, niter_opt, discarded_samples, discarded_samples_on_init,
         method, diagshift, rescale_shift, use_iterative, use_cholesky);

    InitOutput(output_file, save_every);
  }

  void Init(int nsamples, int niter_opt, int discarded_samples,
            int discarded_samples_on_init, const std::string &method,
            double diagshift, bool rescale_shift, bool use_iterative,
            bool use_cholesky) {
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

    niter_opt_ = niter_opt;

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

  void InitOutput(std::string filebase, int freqbackup) {
    if (mynode_ == 0) {
      output_ =
          JsonOutputWriter(filebase + ".log", filebase + ".wf", freqbackup);
    }
  }

  void AddObservable(Operator ob, const std::string &obname) {
    obs_.push_back(ob);
    obsnames_.push_back(obname);
  }

  void InitSweeps() {
    sampler_.SetMachineParameters(psi_.GetParameters());
    sampler_.Reset();

    for (int i = 0; i < ninitsamples_; i++) {
      sampler_.Sweep();
    }
  }

  void Sample() {
    sampler_.SetMachineParameters(psi_.GetParameters());
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

  void Gradient() {
    obsmanager_.Reset("Energy");
    obsmanager_.Reset("EnergyVariance");

    for (const auto &obname : obsnames_) {
      obsmanager_.Reset(obname);
    }

    const int nsamp = vsamp_.rows();
    elocs_.resize(nsamp);
    Ok_.resize(nsamp, psi_.Npar());

    for (int i = 0; i < nsamp; i++) {
      elocs_(i) = Eloc(vsamp_.row(i));
      Ok_.row(i) = psi_.DerLog(vsamp_.row(i));
      obsmanager_.Push("Energy", elocs_(i).real());

      for (std::size_t on = 0; on < obs_.size(); on++) {
        obsmanager_.Push(obsnames_[on], ObSamp(obs_[on], vsamp_.row(i)));
      }
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

  std::complex<double> Eloc(const Eigen::VectorXd &v) {
    ham_.FindConn(v, mel_, connectors_, newconfs_);

    assert(connectors_.size() == mel_.size());

    auto logvaldiffs = (psi_.LogValDiff(v, connectors_, newconfs_));

    assert(mel_.size() == std::size_t(logvaldiffs.size()));

    std::complex<double> eloc = 0;

    for (int i = 0; i < logvaldiffs.size(); i++) {
      eloc += mel_[i] * std::exp(logvaldiffs(i));
    }

    return eloc;
  }

  double ObSamp(const Operator &ob, const Eigen::VectorXd &v) {
    ob.FindConn(v, mel_, connectors_, newconfs_);

    assert(connectors_.size() == mel_.size());

    auto logvaldiffs = (psi_.LogValDiff(v, connectors_, newconfs_));

    assert(mel_.size() == std::size_t(logvaldiffs.size()));

    std::complex<double> obval = 0;

    for (int i = 0; i < logvaldiffs.size(); i++) {
      obval += mel_[i] * std::exp(logvaldiffs(i));
    }

    return obval.real();
  }

  double ElocMean() { return elocmean_.real(); }

  double Elocvar() { return elocvar_; }

  void Run() {
    opt_.Reset();

    InitSweeps();

    for (int i = 0; i < niter_opt_; i++) {
      Sample();

      Gradient();

      UpdateParameters();

      PrintOutput(i);
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
          std::complex<double> nor = (deltaP.dot(S * deltaP));
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

  void PrintOutput(int i) {
    // Note: This has to be called in all MPI processes, because converting
    // the ObsManager to JSON performs a MPI reduction.
    auto obs_data = json(obsmanager_);
    obs_data["Acceptance"] = sampler_.Acceptance();

    if (output_.has_value()) {  // output_.has_value() iff the MPI rank is 0, so
                                // the output is only written once
      output_->WriteLog(i, obs_data);
      output_->WriteState(i, psi_);
    }
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
};

}  // namespace netket

#endif
