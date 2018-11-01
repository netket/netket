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

#ifndef NETKET_SUPERVISEDVARIATIONALMONTECARLO_HPP
#define NETKET_SUPERVISEDVARIATIONALMONTECARLO_HPP

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "Machine/machine.hpp"
#include "Observable/observable.hpp"
#include "Optimizer/optimizer.hpp"
#include "Sampler/sampler.hpp"
#include "Stats/stats.hpp"
#include "Utils/data_utils.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Variational Monte Carlo schemes for supervised learning state given
// Available methods:
// 1) Stochastic reconfiguration optimizer
//   both direct and sparse version
// 2) Gradient Descent optimizer
class SupervisedVariationalMonteCarlo {
  using GsType = std::complex<double>;

  using VectorT =
      Eigen::Matrix<typename Machine<GsType>::StateType, Eigen::Dynamic, 1>;
  using MatrixT = Eigen::Matrix<typename Machine<GsType>::StateType,
                                Eigen::Dynamic, Eigen::Dynamic>;

  using DataType = Data<double>;
  DataType &data_;

  Sampler<Machine<GsType>> &sampler_;
  Machine<GsType> &psi_;

  std::vector<std::vector<int>> connectors_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<std::complex<double>> mel_;

  Eigen::VectorXcd ratios_;
  MatrixT Ok_;
  VectorT Okmean_;
  Eigen::VectorXcd psi_log_amps_;
  Eigen::VectorXcd phi_log_amps_;

  Eigen::MatrixXd vsamp_;

  Eigen::VectorXcd grad_;
  Eigen::VectorXcd gradprev_;

  double sr_diag_shift_;
  bool sr_rescale_shift_;
  bool use_iterative_;

  int totalnodes_;
  int mynode_;

  std::ofstream filelog_;
  std::string filewfname_;
  int freqbackup_;

  Optimizer &opt_;

  Observables obs_;
  ObsManager obsmanager_;
  json outputjson_;

  bool dosr_;

  int nsamples_;
  int nsamples_node_;
  int ninitsamples_;
  int ndiscardedsamples_;
  int niter_opt_;

  std::complex<double> ratio_mean_;
  double elocvar_;
  int npar_;

 public:
  // JSON constructor
  SupervisedVariationalMonteCarlo(DataType &data,
                                  Sampler<Machine<GsType>> &sampler,
                                  Optimizer &opt, const json &pars)
      : data_(data),
        sampler_(sampler),
        psi_(sampler.Psi()),
        opt_(opt),
        obs_(data.GetHilbert(), pars),
        elocvar_(0.) {
    if (FieldExists(pars, "Supervised")) {
      Init(pars);
    } else {
      std::cout << "error in Supervised VMC initialization\n";
    }
  }

  void Init(const json &pars) {
    npar_ = psi_.Npar();

    opt_.Init(psi_.GetParameters());

    grad_.resize(npar_);
    Okmean_.resize(npar_);

    freqbackup_ = 0;

    setSrParameters();

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    nsamples_ = FieldVal(pars["Supervised"], "Nsamples", "Supervised");

    nsamples_node_ = int(std::ceil(double(nsamples_) / double(totalnodes_)));

    ninitsamples_ =
        FieldOrDefaultVal(pars["Supervised"], "DiscardedSamplesOnInit", 0.);

    ndiscardedsamples_ = FieldOrDefaultVal(
        pars["Supervised"], "DiscardedSamples", 0.1 * nsamples_node_);

    niter_opt_ = FieldVal(pars["Supervised"], "NiterOpt", "Supervised");

    std::string file_base =
        FieldVal(pars["Supervised"], "OutputFile", "Supervised");

    int freqbackup = FieldOrDefaultVal(pars["Supervised"], "SaveEvery", 50);
    SetOutName(file_base, freqbackup);

    if (pars["Supervised"]["Method"] == "Gd") {
      dosr_ = false;
    } else {
      double diagshift =
          FieldOrDefaultVal(pars["Supervised"], "DiagShift", 0.01);
      bool rescale_shift =
          FieldOrDefaultVal(pars["Supervised"], "RescaleShift", false);
      bool use_iterative =
          FieldOrDefaultVal(pars["Supervised"], "UseIterative", false);

      setSrParameters(diagshift, rescale_shift, use_iterative);
    }

    if (dosr_) {
      InfoMessage() << "Using the Stochastic reconfiguration method"
                    << std::endl;
      if (use_iterative_) {
        InfoMessage() << "With iterative solver" << std::endl;
      }
    } else {
      InfoMessage() << "Using a gradient-descent based method" << std::endl;
    }

    InfoMessage() << "Variational Monte Carlo running on " << totalnodes_
                  << " processes" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
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

  // Sets the name of the files on which the logs and the wave-function
  // parameters are saved the wave-function is saved every freq steps
  void SetOutName(const std::string &filebase, double freq = 50) {
    filelog_.open(filebase + std::string(".log"));
    freqbackup_ = freq;

    filewfname_ = filebase + std::string(".wf");
  }

  void Gradient() {
    // Gradient is consisted of three parts:
    // <Ok>, <ratio * Ok>, <ratio>
    obsmanager_.Reset("Ratio");
    obsmanager_.Reset("RatioVariance");

    for (std::size_t i = 0; i < obs_.Size(); i++) {
      obsmanager_.Reset(obs_(i).Name());
    }

    const int nsamp = vsamp_.rows();
    ratios_.resize(nsamp);
    psi_log_amps_.resize(nsamp);
    phi_log_amps_.resize(nsamp);
    Ok_.resize(nsamp, psi_.Npar());

    for (int i = 0; i < nsamp; i++) {
      psi_log_amps_(i) = psi_.LogVal(vsamp_.row(i));
      phi_log_amps_(i) = data_.logVal(vsamp_.row(i));
    }
    auto psi_log_amps_max_ = psi_log_amps_.real().maxCoeff();
    psi_log_amps_ -= psi_log_amps_max_ * Eigen::VectorXd::Ones(nsamp);

    for (int i = 0; i < nsamp; i++) {
      ratios_(i) = std::exp(phi_log_amps_(i) - psi_log_amps_(i));
      // ratios_(i) = Ratio(vsamp_.row(i));
      obsmanager_.Push("Ratio", ratios_(i).real());
      Ok_.row(i) = psi_.DerLog(vsamp_.row(i));

      for (std::size_t k = 0; k < obs_.Size(); k++) {
        obsmanager_.Push(obs_(k).Name(), ObSamp(obs_(k), vsamp_.row(i)));
      }
    }

    ratio_mean_ = ratios_.mean();
    SumOnNodes(ratio_mean_);
    ratio_mean_ /= double(totalnodes_);

    Okmean_ = Ok_.colwise().mean();
    SumOnNodes(Okmean_);
    Okmean_ /= double(totalnodes_);

    Ok_ = Ok_.rowwise() - Okmean_.transpose();

    ratios_ -= ratio_mean_ * Eigen::VectorXd::Ones(nsamp);

    for (int i = 0; i < nsamp; i++) {
      obsmanager_.Push("RatioVariance", std::norm(ratios_(i)));
    }

    // grad_ = 2. * (Ok_.adjoint() * ratios_);
    grad_ = -(Ok_.adjoint() * ratios_) / ratio_mean_;

    // Summing the gradient over the nodes
    SumOnNodes(grad_);
    grad_ /= double(totalnodes_ * nsamp);
  }

  std::complex<double> Ratio(const Eigen::VectorXd &v) {
    auto log_amp_psi = psi_.LogVal(v);
    auto log_amp_phi = data_.logVal(v);
    std::complex<double> log_amp_diff = log_amp_phi - log_amp_psi;
    return std::exp(log_amp_diff);
  }

  double ObSamp(Observable &ob, const Eigen::VectorXd &v) {
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

  std::complex<double> RatioMean() { return ratio_mean_; }

  double Elocvar() { return elocvar_; }

  void Run_Supervised() {
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

      Eigen::VectorXcd b = Ok_.adjoint() * ratios_;
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

        Eigen::FullPivHouseholderQR<Eigen::MatrixXcd> qr(S.rows(), S.cols());
        qr.setThreshold(1.0e-6);
        qr.compute(S);
        const Eigen::VectorXcd deltaP = qr.solve(b);
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
    auto Acceptance = sampler_.Acceptance();

    auto jiter = json(obsmanager_);
    jiter["Iteration"] = i;
    outputjson_["Output"].push_back(jiter);

    if (mynode_ == 0) {
      if (jiter["Iteration"] != 0) {
        long pos = filelog_.tellp();
        filelog_.seekp(pos - 3);
        filelog_.write(",  ", 3);
        filelog_ << jiter << "]}" << std::endl;
      } else {
        filelog_ << outputjson_ << std::endl;
      }
    }

    if (mynode_ == 0 && freqbackup_ > 0 && std::fmod(i, freqbackup_) < 0.5) {
      psi_.Save(filewfname_);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void setSrParameters(double diagshift = 0.01, bool rescale_shift = false,
                       bool use_iterative = false) {
    sr_diag_shift_ = diagshift;
    sr_rescale_shift_ = rescale_shift;
    use_iterative_ = use_iterative;
    dosr_ = true;
  }
};

}  // namespace netket

#endif
