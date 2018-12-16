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

#ifndef NETKET_QUANTUMSTATERECONSTRUCTION_HPP_
#define NETKET_QUANTUMSTATERECONSTRUCTION_HPP_

#include <bitset>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "Machine/machine.hpp"
#include "Operator/abstract_operator.hpp"
#include "Optimizer/optimizer.hpp"
#include "Output/json_output_writer.hpp"
#include "Sampler/abstract_sampler.hpp"
#include "Stats/stats.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

class QuantumStateReconstruction {
  using GsType = std::complex<double>;
  using VectorT = Eigen::Matrix<typename AbstractMachine<GsType>::StateType,
                                Eigen::Dynamic, 1>;
  using MatrixT = Eigen::Matrix<typename AbstractMachine<GsType>::StateType,
                                Eigen::Dynamic, Eigen::Dynamic>;

  AbstractSampler<AbstractMachine<GsType>> &sampler_;
  AbstractMachine<GsType> &psi_;
  const AbstractHilbert &hilbert_;
  AbstractOptimizer &opt_;

  std::vector<LocalOperator> rotations_;

  std::vector<std::vector<int>> connectors_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<std::complex<double>> mel_;

  MatrixT Ok_;
  VectorT Okmean_;

  Eigen::MatrixXd vsamp_;
  Eigen::VectorXcd grad_;
  Eigen::VectorXcd rotated_grad_;

  // This optional will contain a value iff the MPI rank is 0.
  nonstd::optional<JsonOutputWriter> output_;

  int totalnodes_;
  int mynode_;

  std::vector<AbstractOperator *> obs_;
  std::vector<std::string> obsnames_;
  ObsManager obsmanager_;

  int batchsize_;
  int batchsize_node_;
  int nsamples_;
  int nsamples_node_;
  int ninitsamples_;
  int ndiscardedsamples_;
  int niter_opt_;

  int npar_;

  DistributedRandomEngine rgen_;

  std::vector<Eigen::VectorXd> trainingSamples_;
  std::vector<int> trainingBases_;

 public:
  using MatType = LocalOperator::MatType;

  QuantumStateReconstruction(
      AbstractSampler<AbstractMachine<GsType>> &sampler, AbstractOptimizer &opt,
      int batchsize, int nsamples, int niter_opt, std::vector<MatType> jrot,
      std::vector<std::vector<int>> sites,
      std::vector<Eigen::VectorXd> trainingSamples,
      std::vector<int> trainingBases, std::string output_file,
      int ndiscardedsamples = -1, int discarded_samples_on_init = 0)
      : sampler_(sampler),
        psi_(sampler_.GetMachine()),
        hilbert_(psi_.GetHilbert()),
        opt_(opt),
        trainingSamples_(trainingSamples),
        trainingBases_(trainingBases) {
    npar_ = psi_.Npar();

    opt_.Init(psi_.GetParameters());

    grad_.resize(npar_);
    rotated_grad_.resize(npar_);

    Okmean_.resize(npar_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    batchsize_ = batchsize;
    batchsize_node_ = int(std::ceil(double(batchsize_) / double(totalnodes_)));

    nsamples_ = nsamples;
    nsamples_node_ = int(std::ceil(double(nsamples_) / double(totalnodes_)));

    ninitsamples_ = discarded_samples_on_init;

    if (ndiscardedsamples == -1) {
      ndiscardedsamples_ = 0.1 * nsamples_node_;
    } else {
      ndiscardedsamples_ = ndiscardedsamples;
    }

    niter_opt_ = niter_opt;

    // TODO Change this hack
    for (std::size_t i = 0; i < jrot.size(); i++) {
      if (sites[i].size() == 0) {
        LocalOperator::SiteType vec(1, 0);
        LocalOperator::MatType id(2);
        id[0].resize(2);
        id[1].resize(2);
        id[0][1] = 0.0;
        id[1][0] = 0.0;
        id[0][0] = 1.0;
        id[1][1] = 1.0;
        rotations_.push_back(LocalOperator(hilbert_, id, vec));
      } else {
        rotations_.push_back(LocalOperator(hilbert_, jrot[i], sites[i]));
      }
    }

    InfoMessage() << "Quantum state reconstruction running on " << totalnodes_
                  << " processes" << std::endl;
    InitOutput(output_file);

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void InitOutput(std::string filebase) {
    if (mynode_ == 0) {
      output_ = JsonOutputWriter(filebase + ".log", filebase + ".wf", 1);
    }
  }

  void InitSweeps() {
    sampler_.Reset();

    for (int i = 0; i < ninitsamples_; i++) {
      sampler_.Sweep();
    }
  }

  void AddObservable(AbstractOperator &ob, const std::string &obname) {
    obs_.push_back(&ob);
    obsnames_.push_back(obname);
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

  void Gradient(std::vector<Eigen::VectorXd> &batchSamples,
                std::vector<int> &batchBases) {
    Eigen::VectorXcd der(psi_.Npar());

    for (const auto &obname : obsnames_) {
      obsmanager_.Reset(obname);
    }

    // Positive phase driven by data
    const int ndata = batchsize_node_;
    Ok_.resize(ndata, psi_.Npar());
    for (int i = 0; i < ndata; i++) {
      RotateGradient(batchBases[i], batchSamples[i], der);
      Ok_.row(i) = der.conjugate();
    }
    grad_ = -2.0 * (Ok_.colwise().mean());

    // Negative phase driven by the machine
    Sample();

    const int nsamp = vsamp_.rows();
    Ok_.resize(nsamp, psi_.Npar());

    for (int i = 0; i < nsamp; i++) {
      Ok_.row(i) = psi_.DerLog(vsamp_.row(i)).conjugate();

      for (std::size_t on = 0; on < obs_.size(); on++) {
        obsmanager_.Push(obsnames_[on], ObSamp(*obs_[on], vsamp_.row(i)));
      }
    }
    grad_ += 2.0 * (Ok_.colwise().mean());

    // Summing the gradient over the nodes
    SumOnNodes(grad_);
    grad_ /= double(totalnodes_);
  }

  double ObSamp(AbstractOperator &ob, const Eigen::VectorXd &v) {
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

  void Run() {
    std::vector<Eigen::VectorXd> batchSamples;
    std::vector<int> batchBases;
    opt_.Reset();

    InitSweeps();
    std::uniform_int_distribution<int> distribution(
        0, trainingSamples_.size() - 1);

    for (int i = 0; i < niter_opt_; i++) {
      int index;
      batchSamples.resize(batchsize_node_);
      batchBases.resize(batchsize_node_);

      // Randomly select a batch of training data
      for (int k = 0; k < batchsize_node_; k++) {
        index = distribution(rgen_.Get());
        batchSamples[k] = trainingSamples_[index];
        batchBases[k] = trainingBases_[index];
      }
      Gradient(batchSamples, batchBases);
      UpdateParameters();
      PrintOutput(i);
    }
  }

  void UpdateParameters() {
    auto pars = psi_.GetParameters();
    opt_.Update(grad_, pars);
    SendToAll(pars);
    psi_.SetParameters(pars);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Evaluate the gradient for a given visible state, in the
  // basis identified by b_index
  void RotateGradient(int b_index, const Eigen::VectorXd &state,
                      Eigen::VectorXcd &rotated_gradient) {
    std::complex<double> den;
    Eigen::VectorXcd num;
    Eigen::VectorXd v(psi_.Nvisible());
    rotations_[b_index].FindConn(state, mel_, connectors_, newconfs_);
    assert(connectors_.size() == mel_.size());

    const std::size_t nconn = connectors_.size();

    const auto logvaldiffs = (psi_.LogValDiff(state, connectors_, newconfs_));
    den = 0.0;
    num.setZero(psi_.Npar());
    for (std::size_t k = 0; k < nconn; k++) {
      v = state;
      for (std::size_t j = 0; j < connectors_[k].size(); j++) {
        v(connectors_[k][j]) = newconfs_[k][j];
      }
      num += mel_[k] * std::exp(logvaldiffs(k)) * psi_.DerLog(v);
      den += mel_[k] * std::exp(logvaldiffs(k));
    }
    if (!std::isfinite(std::abs(den))) {
      std::cout << den << std::endl;
      for (std::size_t k = 0; k < nconn; k++) {
        v = state;
        for (std::size_t j = 0; j < connectors_[k].size(); j++) {
          v(connectors_[k][j]) = newconfs_[k][j];
        }
        std::cout << mel_[k] << " <-mel" << std::endl;
        std::cout << psi_.DerLog(v) << " <-derlog" << std::endl;
        std::cout << logvaldiffs(k) << " <-logvaldiffs" << std::endl;
        std::cout << std::exp(logvaldiffs(k)) << " <-explog" << std::endl;
      }
      std::exit(1);
    }
    rotated_gradient = (num / den);
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
};

}  // namespace netket

#endif  // NETKET_UNSUPERVISED_HPP
