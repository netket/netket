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
#include "Output/json_output_writer.hpp"
#include "Sampler/abstract_sampler.hpp"
#include "Stats/stats.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

class Supervised {
  using complex = std::complex<double>;

  AbstractSampler<AbstractMachine<complex>> &sampler_;
  AbstractMachine<complex> &psi_;
  AbstractOptimizer &opt_;

  // Total batchsize
  int batchsize_;
  // Batchsize per node
  int batchsize_node_;
  // Number of epochs for training
  int niter_opt_ = 10;

  // Total number of computational nodes to run on
  int totalnodes_;
  int mynode_;

  // Number of parameters of the machine
  int npar_;
  // Stores the gradient of loss w.r.t. the machine parameters
  Eigen::VectorXcd grad_;

  // Training samples and targets
  std::vector<Eigen::VectorXd> trainingSamples_;
  std::vector<Eigen::VectorXd> trainingTargets_;
  // Test samples and targets
  std::vector<Eigen::VectorXd> testSamples_;
  std::vector<Eigen::VectorXd> testTargets_;

  // Random number generator
  netket::default_random_engine rgen_;

 public:
  Supervised(AbstractSampler<AbstractMachine<complex>> &sampler,
             AbstractOptimizer &opt, int batchsize, int niter_opt,
             std::vector<Eigen::VectorXd> trainingSamples,
             std::vector<Eigen::VectorXd> trainingTargets,
             std::string output_file)
      : sampler_(sampler),
        psi_(sampler_.GetMachine()),
        opt_(opt),
        trainingSamples_(trainingSamples),
        trainingTargets_(trainingTargets) {
    npar_ = psi_.Npar();

    opt_.Init(psi_.GetParameters());

    grad_.resize(npar_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    batchsize_ = batchsize;
    batchsize_node_ = int(std::ceil(double(batchsize_) / double(totalnodes_)));

    niter_opt_ = niter_opt;

    InfoMessage() << "Supervised learning running on " << totalnodes_
                  << " processes" << std::endl;
    InitOutput(output_file);

    MPI_Barrier(MPI_COMM_WORLD);
  }

  /// Initializes the output log and wavefunction files
  void InitOutput(std::string filebase) {
    // Only the first node
    if (mynode_ == 0) {
      JsonOutputWriter(filebase + ".log", filebase + ".wf", 1);
    }
  }

  /// Computes the derivative of negative log of wavefunction overlap,
  /// taken from https://arxiv.org/abs/1808.05232
  void LogOverlap(std::vector<Eigen::VectorXd> &batchSamples,
                  std::vector<Eigen::VectorXd> &batchTargets) {
    // Allocate vectors for storing the derivatives ...
    Eigen::VectorXcd num1(psi_.Npar());
    Eigen::VectorXcd num2(psi_.Npar());
    Eigen::VectorXcd num3(psi_.Npar());
    complex den(0.0, 0.0);
    Eigen::VectorXcd total_der(psi_.Npar());

    // ... and zero them out
    num1.setZero(psi_.Npar());
    num2.setZero(psi_.Npar());
    num3.setZero(psi_.Npar());
    total_der.setZero(psi_.Npar());

    // Foreach sample in the batch
    for (int i = 0; i < batchsize_node_; i++) {
      // Extract log(config)
      Eigen::VectorXd sample(batchSamples[i]);
      // And the corresponding target
      Eigen::VectorXd target(batchTargets[i]);

      // Cast value and target to std::complex<couble>
      complex value(psi_.LogVal(sample));
      complex t(target[0], target[1]);
      auto der = psi_.DerLog(sample);

      num1 = num1 + der * pow(abs(value), 2);
      num2 = num2 + der * pow(abs(value), 2) * target / value;
      num3 = num3 + pow(abs(value), 2) * target / value;
      den = den + pow(abs(value), 2);

      total_der = total_der + (num1 / den) - num2 * num3.inverse();
    }
    // Store derivatives in grad_ ...
    grad_ = total_der;

    // ... and compute the mean of the gradient over the nodes
    SumOnNodes(grad_);
    grad_ /= double(totalnodes_);
  }

  /// Computes the gradient of the loss function with respect to
  /// the machine's parameters for a given batch of samples and targets
  /// TODO(everthmore): User defined loss function instead of hardcoded MSE
  void Gradient(std::vector<Eigen::VectorXd> &batchSamples,
                std::vector<Eigen::VectorXd> &batchTargets) {
    // Allocate a vector for storing the derivatives ...
    Eigen::VectorXcd der(psi_.Npar());
    // ... and zero it out
    der.setZero(psi_.Npar());

    // Foreach sample in the batch
    for (int i = 0; i < batchsize_node_; i++) {
      // Extract log(config)
      Eigen::VectorXd sample(batchSamples[i]);
      // And the corresponding target
      Eigen::VectorXd target(batchTargets[i]);

      // Cast value and target to std::complex<couble>
      complex value(psi_.LogVal(sample));
      complex t(target[0], target[1]);

      /// TODO(everthemore): We need to decide whether the user needs
      ///                    to provide logvals of targets, or whether we do
      ///                    so here

      auto partial_gradient = psi_.DerLog(sample);

      // MSE loss
      der = der + partial_gradient * (value - t);
    }
    // Store derivatives in grad_ ...
    grad_ = der;
    // ... and compute the mean of the gradient over the nodes
    SumOnNodes(grad_);
    grad_ /= double(totalnodes_);
  }

  /// Runs the supervised learning on the training samples and targets
  /// TODO(everthmore): Override w/ function call that sets testSamples_
  ///                   and testTargets_ and reports on accuracy on those.
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
      LogOverlap(batchSamples, batchTargets);
      UpdateParameters();
      PrintMSE();
    }
  }

  /// Updates the machine parameters with the current gradient
  void UpdateParameters() {
    auto pars = psi_.GetParameters();
    opt_.Update(grad_, pars);
    SendToAll(pars);
    psi_.SetParameters(pars);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /// Outputs the current Mean-Squared-Error (mostly debugging, temporarily)
  void PrintMSE() {
    const int numSamples = trainingSamples_.size();

    std::complex<double> mse = 0.0;
    for (int i = 0; i < numSamples; i++) {
      Eigen::VectorXd sample = trainingSamples_[i];
      Eigen::VectorXd target = trainingTargets_[i];

      complex value(psi_.LogVal(sample));
      complex t(target[0], target[1]);

      mse += 0.5 * pow(value - t, 2);
    }

    std::cout << "MSE: " << mse << std::endl;
  }
};

}  // namespace netket

#endif
