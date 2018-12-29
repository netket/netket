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

#ifndef NETKET_SUPERVISED_CC
#define NETKET_SUPERVISED_CC

#include <bitset>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "Machine/machine.hpp"
#include "Optimizer/optimizer.hpp"
#include "Sampler/abstract_sampler.hpp"
#include "Stats/stats.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

class Supervised {
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

  using GsType = std::complex<double>;

  AbstractSampler<AbstractMachine<GsType>> &sampler_;
  AbstractMachine<GsType> &psi_;
  AbstractOptimizer &opt_;

  Eigen::VectorXcd grad_;

  int batchsize_ = 100;
  // Batchsize per node
  int batchsize_node_ = 100;

  // Number of parameters of the machine
  int npar_;

  // Number of epochs for training
  int niter_opt_ = 10;

  std::vector<Eigen::VectorXd> trainingSamples_;
  std::vector<Eigen::VectorXd> trainingTargets_;

  netket::default_random_engine rgen_;

  Eigen::MatrixXd inputs_;
  Eigen::VectorXcd targets_;

 public:
  Supervised(AbstractSampler<AbstractMachine<GsType>> &sampler,
             AbstractOptimizer &opt, int batchsize, int niter_opt,
             std::vector<Eigen::VectorXd> trainingSamples,
             std::vector<Eigen::VectorXd> trainingTargets,
             std::string output_file)
      : sampler_(sampler),
        psi_(sampler_.GetMachine()),
        opt_(opt),
        trainingSamples_(trainingSamples),
        trainingTargets_(trainingTargets) {
    batchsize_ = batchsize;
    niter_opt_ = niter_opt;

    npar_ = psi_.Npar();

    opt_.Init(psi_.GetParameters());

    grad_.resize(npar_);

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void Gradient(std::vector<Eigen::VectorXd> &batchSamples,
                std::vector<Eigen::VectorXd> &batchTargets) {
    // Vector for storing the derivatives
    Eigen::VectorXcd der(psi_.Npar());

    std::complex<double> t;

    // Foreach sample in the batch
    const int ndata = batchsize_node_;
    der.setZero(psi_.Npar());
    for (int i = 0; i < ndata; i++) {
      // Extract log(config)
      Eigen::VectorXd sample(batchSamples[i]);
      std::complex<double> value = psi_.LogVal(sample);

      // And the corresponding target
      Eigen::VectorXd target(batchTargets[i]);
      t.real(target[0]);
      t.imag(target[1]);

      auto partial_gradient = psi_.DerLog(sample);
      der = der + partial_gradient * (value - t);
    }
    grad_ = der;

    // Summing the gradient over the nodes
    SumOnNodes(grad_);
    // grad_ /= double(totalnodes_);
  }

  void Run() {
    std::vector<Eigen::VectorXd> batchSamples;
    std::vector<Eigen::VectorXd> batchTargets;

    opt_.Reset();

    // Initialize a uniform distribution to draw training samples from
    std::uniform_int_distribution<int> distribution(
        0, trainingSamples_.size() - 1);

    for (int i = 0; i < niter_opt_; i++) {
      int index;
      batchSamples.resize(batchsize_node_);
      batchTargets.resize(batchsize_node_);

      // Randomly select a batch of training data
      for (int k = 0; k < batchsize_node_; k++) {
        // Draw from the distribution using the netket random number generator
        index = distribution(rgen_);
        batchSamples[k] = trainingSamples_[index];
        batchTargets[k] = trainingTargets_[index];
      }

      // Compute the gradient on the batch samples
      Gradient(batchSamples, batchTargets);
      UpdateParameters();
      PrintMSE();

      // std::cout << " grad norm " << grad_.norm() << std::endl;
    }
  }

  void UpdateParameters() {
    auto pars = psi_.GetParameters();
    opt_.Update(grad_, pars);
    SendToAll(pars);
    psi_.SetParameters(pars);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void PrintMSE() {
    const int numSamples = trainingSamples_.size();

    std::complex<double> t;
    std::complex<double> value;

    std::complex<double> mse = 0.0;
    for (int i = 0; i < numSamples; i++) {
      Eigen::VectorXd sample = trainingSamples_[i];
      Eigen::VectorXd target = trainingTargets_[i];

      value = psi_.LogVal(sample);
      t.real(target[0]);
      t.imag(target[1]);

      mse += pow(value - t, 2);
    }

    std::cout << "MSE: " << mse << std::endl;
  }
};

}  // namespace netket

#endif
