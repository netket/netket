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
#include <limits>
#include <string>
#include <vector>
#include "Machine/machine.hpp"
#include "Optimizer/optimizer.hpp"
#include "Output/json_output_writer.hpp"
#include "Stats/stats.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

class Supervised {
  using MatrixT = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

  AbstractMachine &psi_;
  AbstractOptimizer &opt_;

  SR sr_;
  bool dosr_;

  // Total batchsize
  int batchsize_;
  // Batchsize per node
  int batchsize_node_;

  // Total number of computational nodes to run on
  int totalnodes_;
  int mynode_;

  // Number of parameters of the machine
  int npar_;
  // Stores the gradient of loss w.r.t. the machine parameters
  Eigen::VectorXcd grad_;
  Eigen::VectorXcd grad_part_1_;
  Eigen::VectorXcd grad_part_2_;
  Complex grad_num_1_;
  Complex grad_num_2_;

  // Training samples and targets
  std::vector<Eigen::VectorXd> trainingSamples_;
  std::vector<Eigen::VectorXcd> trainingTargets_;
  // Test samples and targets
  std::vector<Eigen::VectorXd> testSamples_;
  std::vector<Eigen::VectorXcd> testTargets_;

  // All loss function is real
  double loss_log_overlap_;
  double loss_mse_;
  double loss_mse_log_;

  std::uniform_int_distribution<int> distribution_uni_;
  std::discrete_distribution<> distribution_phi_;

  MatrixT Ok_;

 protected:
  // Random number generator with correct seeding for parallel processes
  default_random_engine &GetRandomEngine() { return engine_.Get(); }

 private:
  DistributedRandomEngine engine_;

 public:
  Supervised(AbstractMachine &psi, AbstractOptimizer &opt, int batchsize,
             std::vector<Eigen::VectorXd> trainingSamples,
             std::vector<Eigen::VectorXcd> trainingTargets,
             const std::string &method = "Gd", double diag_shift = 0.01,
             bool use_iterative = false, bool use_cholesky = true)
      : psi_(psi),
        opt_(opt),
        trainingSamples_(trainingSamples),
        trainingTargets_(trainingTargets) {
    npar_ = psi_.Npar();

    opt_.Init(npar_, psi_.IsHolomorphic());

    grad_.resize(npar_);
    grad_part_1_.resize(npar_);
    grad_part_2_.resize(npar_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    batchsize_ = batchsize;
    batchsize_node_ = int(std::ceil(double(batchsize_) / double(totalnodes_)));

    // Initialize a uniform distribution to draw training samples from
    distribution_uni_ =
        std::uniform_int_distribution<int>(0, trainingSamples_.size() - 1);

    std::vector<double> trainingTarget_values_;
    trainingTarget_values_.resize(trainingTargets_.size());
    for (unsigned int i = 0; i < trainingTargets_.size(); i++) {
      trainingTarget_values_[i] = exp(2 * trainingTargets_[i][0].real());
    }
    distribution_phi_ = std::discrete_distribution<>(
        trainingTarget_values_.begin(), trainingTarget_values_.end());

    if (method == "Gd") {
      dosr_ = false;
      InfoMessage() << "Using a gradient-descent based method" << std::endl;
    } else {
      setSrParameters(diag_shift, use_iterative, use_cholesky);
    }

    InfoMessage() << "Supervised learning running on " << totalnodes_
                  << " processes" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
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

    double max_log_psi = 0;
    /// [TODO] avoid going through psi twice.
    for (int i = 0; i < batchsize_node_; i++) {
      Complex value(psi_.LogVal(batchSamples[i]));
      if (max_log_psi < value.real()) {
        max_log_psi = value.real();
      }
    }

    Ok_.resize(batchsize_node_, psi_.Npar());

    // For each sample in the batch
    for (int i = 0; i < batchsize_node_; i++) {
      // Extract log(config)
      Eigen::VectorXd sample(batchSamples[i]);
      // And the corresponding target
      Eigen::VectorXcd target(batchTargets[i]);
      Complex t(target[0].real(), target[0].imag());
      // Undo log
      t = exp(t);

      Complex value(psi_.LogVal(sample));
      // Undo Log
      value = value - max_log_psi;
      value = exp(value);

      // Compute derivative of log
      auto der = psi_.DerLog(sample);
      Ok_.row(i) = der;

      der = der.conjugate();

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

    double max_log_psi = -std::numeric_limits<double>::infinity();
    /// [TODO] avoid going through psi twice.
    for (int i = 0; i < batchsize_node_; i++) {
      Complex value(psi_.LogVal(batchSamples[i]));
      if (max_log_psi < value.real()) {
        max_log_psi = value.real();
      }
    }

    Ok_.resize(batchsize_node_, psi_.Npar());

    // For each sample in the batch
    for (int i = 0; i < batchsize_node_; i++) {
      // Extract log(config)
      Eigen::VectorXd sample(batchSamples[i]);
      // And the corresponding target
      Eigen::VectorXcd target(batchTargets[i]);
      Complex t = target[0];
      // Undo log
      t = exp(t);

      Complex value(psi_.LogVal(sample));
      // Undo Log
      value = value - max_log_psi;
      value = exp(value);

      // Compute derivative of log
      auto der = psi_.DerLog(sample);
      Ok_.row(i) = der;
      der = der.conjugate();

      grad_part_1_ = grad_part_1_ + der * std::norm(value / t);
      grad_num_1_ = grad_num_1_ + std::norm(value / t);
      grad_part_2_ = grad_part_2_ + der * std::conj(value / t);
      grad_num_2_ = grad_num_2_ + std::conj(value / t);
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
  void GradientComplexMse(std::vector<Eigen::VectorXd> &batchSamples,
                          std::vector<Eigen::VectorXcd> &batchTargets) {
    // Allocate a vector for storing the derivatives ...
    Eigen::VectorXcd der(psi_.Npar());
    // ... and zero it out
    der.setZero(psi_.Npar());

    Ok_.resize(batchsize_node_, psi_.Npar());

    // Foreach sample in the batch
    for (int i = 0; i < batchsize_node_; i++) {
      // Extract complex value of log(config)
      Eigen::VectorXd sample(batchSamples[i]);
      Complex value(psi_.LogVal(sample));

      // And the corresponding target
      Eigen::VectorXcd target(batchTargets[i]);
      Complex t(target[0].real(), target[0].imag());

      // Compute derivative of log(psi)
      auto partial_gradient = psi_.DerLog(sample);
      Ok_.row(i) = partial_gradient;

      // MSE loss
      der = der + (partial_gradient.conjugate()) * (value - t);
    }
    // Store derivatives in grad_ ...
    grad_ = der / batchsize_node_;

    // ... and compute the mean of the gradient over the nodes
    SumOnNodes(grad_);
    grad_ /= double(totalnodes_);
  }

  void Advance(std::string lossFunction) {
    std::vector<Eigen::VectorXd> batchSamples(batchsize_node_);
    std::vector<Eigen::VectorXcd> batchTargets(batchsize_node_);

    int index;

    if (lossFunction == "MSE" || lossFunction == "Overlap_uni") {
      // Randomly select a batch of training data
      for (int k = 0; k < batchsize_node_; k++) {
        // Draw from the distribution using the netket random number generator
        index = distribution_uni_(this->GetRandomEngine());
        batchSamples[k] = trainingSamples_[index];
        batchTargets[k] = trainingTargets_[index];
      }
    } else if (lossFunction == "Overlap_phi") {
      // Randomly select a batch of training data
      for (int k = 0; k < batchsize_node_; k++) {
        // Draw from the distribution using the netket random number generator
        index = distribution_phi_(this->GetRandomEngine());
        batchSamples[k] = trainingSamples_[index];
        batchTargets[k] = trainingTargets_[index];
      }
    } else {
      std::cout << "Supervised loss function \" " << lossFunction
                << "\" undefined!" << std::endl;
    }

    if (lossFunction == "MSE") {
      GradientComplexMse(batchSamples, batchTargets);
      UpdateParameters();
      ComputeLosses();
    } else if (lossFunction == "Overlap_uni") {
      DerLogOverlap_uni(batchSamples, batchTargets);
      UpdateParameters();
      ComputeLosses();
    } else if (lossFunction == "Overlap_phi") {
      DerLogOverlap_phi(batchSamples, batchTargets);
      UpdateParameters();
      ComputeLosses();
    } else {
      std::cout << "Supervised loss function \" " << lossFunction
                << "\" undefined!" << std::endl;
    }
  }

  /// Runs the supervised learning on the training samples and targets
  /// TODO(everthmore): Override w/ function call that sets testSamples_
  ///                   and testTargets_ and reports on accuracy on those.
  void Run(int n_iter, const std::string &lossFunction = "MSE",
           const std::string &output_prefix = "output",
           int save_params_every = 50) {
    assert(n_iter > 0);
    assert(save_params_every > 0);

    /// Writer to the output
    /// This optional will contain a value iff the MPI rank is 0.
    nonstd::optional<JsonOutputWriter> writer;
    if (mynode_ == 0) {
      /// Initializes the output log and wavefunction files
      writer.emplace(output_prefix + ".log", output_prefix + ".wf",
                     save_params_every);
    }

    opt_.Reset();
    for (int i = 0; i < n_iter; i++) {
      Advance(lossFunction);
      // writer.has_value() iff the MPI rank is 0, so the output is only
      // written once
      if (writer.has_value()) {
        json out_data;
        out_data["log_overlap"] = loss_log_overlap_;
        out_data["mse"] = loss_mse_;
        out_data["mse_log"] = loss_mse_log_;

        writer->WriteLog(i, out_data);
        writer->WriteState(i, psi_);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  /// Updates the machine parameters with the current gradient
  void UpdateParameters() {
    auto pars = psi_.GetParameters();

    Eigen::VectorXcd deltap(npar_);

    if (dosr_) {
      sr_.ComputeUpdate(Ok_, grad_, deltap);
    } else {
      deltap = grad_;
    }

    opt_.Update(deltap, pars);
    SendToAll(pars);

    psi_.SetParameters(pars);
  }

  void ComputeLosses() {
    ComputeMse();
    ComputeLogOverlap();
  }

  /// Compute the MSE of psi and the MSE of log(psi)
  /// for monitoring the convergence.
  void ComputeMse() {
    const int numSamples = trainingSamples_.size();

    double mse_log = 0.0;
    double mse = 0.0;
    for (int i = 0; i < numSamples; i++) {
      Eigen::VectorXd sample = trainingSamples_[i];
      Complex value(psi_.LogVal(sample));

      Eigen::VectorXcd target = trainingTargets_[i];
      Complex t(target[0].real(), target[0].imag());

      mse_log += 0.5 * std::norm(value - t);
      mse += 0.5 * std::norm(exp(value) - exp(t));
    }

    loss_mse_ = mse / numSamples;
    loss_mse_log_ = mse_log / numSamples;
  }

  double GetMse() const { return loss_mse_; }

  double GetMseLog() const { return loss_mse_log_; }

  void ComputeLogOverlap() {
    const int numSamples = trainingSamples_.size();

    // Allocate vectors for storing the derivatives ...
    Complex num1(0.0, 0.0);
    Complex num2(0.0, 0.0);
    Complex num3(0.0, 0.0);
    Complex num4(0.0, 0.0);

    Eigen::VectorXcd logpsi(numSamples);
    double max_log_psi = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < numSamples; i++) {
      logpsi(i) = psi_.LogVal(trainingSamples_[i]);
      if (std::real(logpsi(i)) > max_log_psi) {
        max_log_psi = std::real(logpsi(i));
      }
    }

    for (int i = 0; i < numSamples; i++) {
      // Extract log(config)
      Eigen::VectorXd sample(trainingSamples_[i]);
      // And the corresponding target
      Eigen::VectorXcd target(trainingTargets_[i]);

      // Cast value and target to Complex and undo logs
      Complex value(psi_.LogVal(sample));
      value = exp(value - max_log_psi);
      Complex t(target[0].real(), target[0].imag());
      t = exp(t);

      num1 += std::conj(value) * t;
      num2 += value * std::conj(t);
      num3 += std::norm(value);
      num4 += std::norm(t);
    }

    Complex complex_log_overlap_ =
        -(log(num1) + log(num2) - log(num3) - log(num4));
    assert(std::abs(complex_log_overlap_.imag()) < 1e-8);
    loss_log_overlap_ = complex_log_overlap_.real();
  }

  double GetLogOverlap() const { return loss_log_overlap_; }

  void setSrParameters(double diag_shift = 0.01, bool use_iterative = false,
                       bool use_cholesky = true) {
    dosr_ = true;
    sr_.setParameters(diag_shift, use_iterative, use_cholesky,
                      psi_.IsHolomorphic());
  }
};

}  // namespace netket

#endif
