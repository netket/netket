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
  Eigen::VectorXcd grad_part_1_;
  Eigen::VectorXcd grad_part_2_;
  complex grad_num_1_;
  complex grad_num_2_;

  // Training samples and targets
  std::vector<Eigen::VectorXd> trainingSamples_;
  std::vector<Eigen::VectorXcd> trainingTargets_;
  // Test samples and targets
  std::vector<Eigen::VectorXd> testSamples_;
  std::vector<Eigen::VectorXcd> testTargets_;

  // Random number generator
  netket::default_random_engine rgen_;

 public:
  Supervised(AbstractSampler<AbstractMachine<complex>> &sampler,
             AbstractOptimizer &opt, int batchsize, int niter_opt,
             std::vector<Eigen::VectorXd> trainingSamples,
             std::vector<Eigen::VectorXcd> trainingTargets,
             std::string output_file)
      : sampler_(sampler),
        psi_(sampler_.GetMachine()),
        opt_(opt),
        trainingSamples_(trainingSamples),
        trainingTargets_(trainingTargets) {
    npar_ = psi_.Npar();

    opt_.Init(psi_.GetParameters());

    grad_.resize(npar_);
    grad_part_1_.resize(npar_);
    grad_part_2_.resize(npar_);

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

  /// Computes the gradient estimate of the derivative of negative log
  /// of wavefunction overlap, with index i sampled from unifrom[1, N]
  void DerLogOverlap_uni(std::vector<Eigen::VectorXd> &batchSamples,
                         std::vector<Eigen::VectorXcd> &batchTargets) {
    // ... and zero them out
    grad_.setZero(psi_.Npar());
    grad_part_1_.setZero(psi_.Npar());
    grad_part_2_.setZero(psi_.Npar());
    grad_num_1_ = 0;
    grad_num_2_ = 0;

    // For each sample in the batch
    for (int i = 0; i < batchsize_node_; i++) {
      // Extract log(config)
      Eigen::VectorXd sample(batchSamples[i]);
      // And the corresponding target
      Eigen::VectorXcd target(batchTargets[i]);
      complex t(target[0].real(), target[0].imag());
      // Undo log
      t = exp(t);

      complex value(psi_.LogVal(sample));
      // Undo Log
      value = exp(value);

      // Compute derivative of log
      auto der = psi_.DerLog(sample);
      der = der.conjugate();

      //grad_part_1_ = grad_part_1_ + der * pow(abs(value), 2) / pow(abs(t), 2);
      //grad_num_1_ = grad_num_1_ + pow(abs(value), 2) / pow(abs(t), 2);
      //grad_part_2_ = grad_part_2_ + der * std::conj(value) / std::conj(t);
      //grad_num_2_ = grad_num_2_ + std::conj(value) / std::conj(t);

      grad_part_1_ = grad_part_1_ + der * pow(abs(value), 2);
      grad_num_1_ = grad_num_1_ + pow(abs(value), 2);
      grad_part_2_ = grad_part_2_ + der * std::conj(value) * t;
      grad_num_2_ = grad_num_2_ + std::conj(value) * t;
    }

    SumOnNodes(grad_part_1_);
    SumOnNodes(grad_num_1_);
    SumOnNodes(grad_part_2_);
    SumOnNodes(grad_num_2_);
    /// No need to devide by totalnodes_
    grad_ = grad_part_1_ / grad_num_1_ - grad_part_2_ / grad_num_2_;
  }

  /// Computes the gradient estimate of the derivative of negative log
  /// of wavefunction overlap, with index Xi sampled from Phi
  void DerLogOverlap_phi(std::vector<Eigen::VectorXd> &batchSamples,
                         std::vector<Eigen::VectorXcd> &batchTargets) {
    // ... and zero them out
    grad_.setZero(psi_.Npar());
    grad_part_1_.setZero(psi_.Npar());
    grad_part_2_.setZero(psi_.Npar());
    grad_num_1_ = 0;
    grad_num_2_ = 0;

    // For each sample in the batch
    for (int i = 0; i < batchsize_node_; i++) {
      // Extract log(config)
      Eigen::VectorXd sample(batchSamples[i]);
      // And the corresponding target
      Eigen::VectorXcd target(batchTargets[i]);
      complex t(target[0].real(), target[0].imag());
      // Undo log
      t = exp(t);

      complex value(psi_.LogVal(sample));
      // Undo Log
      value = exp(value);

      // Compute derivative of log
      auto der = psi_.DerLog(sample);
      der = der.conjugate();

      grad_part_1_ = grad_part_1_ + der * pow(abs(value), 2) / pow(abs(t), 2);
      grad_num_1_ = grad_num_1_ + pow(abs(value), 2) / pow(abs(t), 2);
      grad_part_2_ = grad_part_2_ + der * std::conj(value) / std::conj(t);
      grad_num_2_ = grad_num_2_ + std::conj(value) / std::conj(t);

      // grad_part_1_ = grad_part_1_ + der * pow(abs(value), 2);
      // grad_num_1_ = grad_num_1_ + pow(abs(value), 2);
      // grad_part_2_ = grad_part_2_ + der * std::conj(value) * t;
      // grad_num_2_ = grad_num_2_ + std::conj(value) * t;
    }

    SumOnNodes(grad_part_1_);
    SumOnNodes(grad_num_1_);
    SumOnNodes(grad_part_2_);
    SumOnNodes(grad_num_2_);
    /// No need to devide by totalnodes_
    grad_ = grad_part_1_ / grad_num_1_ - grad_part_2_ / grad_num_2_;
  }


  /// Computes the gradient of the loss function with respect to
  /// the machine's parameters for a given batch of samples and targets
  /// TODO(everthmore): User defined loss function instead of hardcoded MSE
  /// Loss = 0.5 * (log(psi) - log(target)) * (log(psi) - log(target)).conj()
  /// Partial Der = Real part of (derlog(psi)*(log(psi) - log(t)).conj()
  void GradientComplexMSE(std::vector<Eigen::VectorXd> &batchSamples,
                          std::vector<Eigen::VectorXcd> &batchTargets) {
    // Allocate a vector for storing the derivatives ...
    Eigen::VectorXcd der(psi_.Npar());
    // ... and zero it out
    der.setZero(psi_.Npar());

    // Foreach sample in the batch
    for (int i = 0; i < batchsize_node_; i++) {
      // Extract complex value of log(config)
      Eigen::VectorXd sample(batchSamples[i]);
      complex value(psi_.LogVal(sample));

      // And the corresponding target
      Eigen::VectorXcd target(batchTargets[i]);
      complex t(target[0].real(), target[0].imag());

      // Compute derivative of log(psi)
      auto partial_gradient = psi_.DerLog(sample);

      // MSE loss
      der = der + (partial_gradient.conjugate()) * (value - t);
    }
    // Store derivatives in grad_ ...
    grad_ = der / batchsize_node_;

    // ... and compute the mean of the gradient over the nodes
    SumOnNodes(grad_);
    grad_ /= double(totalnodes_);
  }

  /// Runs the supervised learning on the training samples and targets
  /// TODO(everthmore): Override w/ function call that sets testSamples_
  ///                   and testTargets_ and reports on accuracy on those.
  void Run(std::string lossFunction = "MSE") {
    std::vector<Eigen::VectorXd> batchSamples;
    std::vector<Eigen::VectorXcd> batchTargets;

    opt_.Reset();

    // Initialize a uniform distribution to draw training samples from
    std::uniform_int_distribution<int> distribution_uni(
        0, trainingSamples_.size() - 1);

    std::vector<double> trainingTarget_values_;
    trainingTarget_values_.resize(trainingTargets_.size());
    for (unsigned int i = 0; i < trainingTargets_.size(); i++) {
      trainingTarget_values_[i] = exp(2 * trainingTargets_[i][0].real());
    }
    std::discrete_distribution<int> distribution_phi(
      trainingTarget_values_.begin(), trainingTarget_values_.end());

    for (int i = 0; i < niter_opt_; i++) {
      int index;
      batchSamples.resize(batchsize_node_);
      batchTargets.resize(batchsize_node_);

      if (lossFunction == "MSE" || lossFunction == "Overlap_uni") {
        // Randomly select a batch of training data
        for (int k = 0; k < batchsize_node_; k++) {
          // Draw from the distribution using the netket random number generator
          index = distribution_uni(rgen_);
          batchSamples[k] = trainingSamples_[index];
          batchTargets[k] = trainingTargets_[index];
        }
      } else if (lossFunction == "Overlap_phi") {
        // Randomly select a batch of training data
        for (int k = 0; k < batchsize_node_; k++) {
          // Draw from the distribution using the netket random number generator
          index = distribution_phi(rgen_);
          batchSamples[k] = trainingSamples_[index];
          batchTargets[k] = trainingTargets_[index];
        }
      } else {
        std::cout << "Supervised loss function \" " << lossFunction
                  << "\" undefined!" << std::endl;
      }



      if (lossFunction == "MSE") {
        GradientComplexMSE(batchSamples, batchTargets);
        UpdateParameters();
        PrintComplexMSE();
        PrintLogOverlap();
      } else if (lossFunction == "Overlap_uni") {
        DerLogOverlap_uni(batchSamples, batchTargets);
        UpdateParameters();
        PrintComplexMSE();
        PrintLogOverlap();
      } else if (lossFunction == "Overlap_phi") {
	DerLogOverlap_phi(batchSamples, batchTargets);
	UpdateParameters();
	PrintComplexMSE();
	PrintLogOverlap();
      } else {
        std::cout << "Supervised loss function \" " << lossFunction
                  << "\" undefined!" << std::endl;
      }
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
  void PrintComplexMSE() {
    const int numSamples = trainingSamples_.size();

    std::complex<double> mse_log = 0.0;
    std::complex<double> mse = 0.0;
    for (int i = 0; i < numSamples; i++) {
      Eigen::VectorXd sample = trainingSamples_[i];
      complex value(psi_.LogVal(sample));

      Eigen::VectorXcd target = trainingTargets_[i];
      complex t(target[0].real(), target[0].imag());

      mse_log += 0.5 * pow(abs(value - t), 2);
      mse += 0.5 * pow(abs(exp(value) - exp(t)), 2);
    }

    complex n(numSamples, 0);
    std::cout << "MSE (log): " << mse_log / n << " MSE : " << mse / n << std::endl;
  }

  /// Outputs the current Mean-Squared-Error (mostly debugging, temporarily)
  void PrintLogOverlap() {
    const int numSamples = trainingSamples_.size();

    // Allocate vectors for storing the derivatives ...
    complex num1(0.0, 0.0);
    complex num2(0.0, 0.0);
    complex num3(0.0, 0.0);
    complex num4(0.0, 0.0);

    std::complex<double> overlap = 0.0;
    for (int i = 0; i < numSamples; i++) {
      // Extract log(config)
      Eigen::VectorXd sample(trainingSamples_[i]);
      // And the corresponding target
      Eigen::VectorXcd target(trainingTargets_[i]);

      // Cast value and target to std::complex<couble> and undo logs
      complex value(psi_.LogVal(sample));
      value = exp(value);
      complex t(target[0].real(), target[0].imag());
      t = exp(t);

      num1 += std::conj(value) * t;
      num2 += value * std::conj(t);
      num3 += pow(abs(value), 2);
      num4 += pow(abs(t), 2);
    }

    overlap = -(log(num1) + log(num2) - log(num3) - log(num4));
    std::cout << "LogOverlap: " << overlap << std::endl;
  }
};

}  // namespace netket

#endif
