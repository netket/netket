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

// authors: Hugo Théveniaut and Fabien Alet

#ifndef NETKET_CUSTOMSAMPLER_HPP
#define NETKET_CUSTOMSAMPLER_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include "Operator/local_operator.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

namespace netket {

// Metropolis sampling using custom moves provided by user
class CustomSampler : public AbstractSampler {
  LocalOperator move_operators_;
  std::vector<double> operatorsweights_;

  // number of visible units
  const int nv_;

  // states of visible units
  Eigen::VectorXd v_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  // Look-up tables
  typename AbstractMachine::LookupType lt_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  int nstates_;
  std::vector<double> localstates_;

 public:
  CustomSampler(AbstractMachine &psi, const LocalOperator &move_operators,
                const std::vector<double> &move_weights = {})
      : AbstractSampler(psi),
        move_operators_(move_operators),
        nv_(GetHilbert().Size()) {
    Init(move_weights);
  }

  void Init(const std::vector<double> &move_weights) {
    CheckMoveOperators(move_operators_);

    if (GetHilbert().Size() != move_operators_.GetHilbert().Size()) {
      throw InvalidInputError(
          "Move operators in CustomSampler act on a different hilbert space "
          "than the Machine");
    }

    if (move_weights.size()) {
      operatorsweights_ = move_weights;

      if (operatorsweights_.size() != move_operators_.Size()) {
        throw InvalidInputError(
            "The custom sampler definition is inconsistent (between "
            "MoveWeights and MoveOperators sizes)");
      }
    } else {  // By default the stochastic operators are drawn uniformly
      operatorsweights_.resize(move_operators_.Size(), 1.0);
    }

    v_.resize(nv_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (!GetHilbert().IsDiscrete()) {
      throw InvalidInputError(
          "Custom Metropolis sampler works only for discrete Hilbert spaces");
    }

    accept_.resize(1);
    moves_.resize(1);

    nstates_ = GetHilbert().LocalSize();
    localstates_ = GetHilbert().LocalStates();

    Reset(true);

    InfoMessage() << "Custom Metropolis sampler is ready " << std::endl;
  }

  void Reset(bool initrandom = false) override {
    if (initrandom) {
      GetHilbert().RandomVals(v_, this->GetRandomEngine());
    }

    GetMachine().InitLookup(v_, lt_);

    accept_ = Eigen::VectorXd::Zero(1);
    moves_ = Eigen::VectorXd::Zero(1);
  }

  void Sweep() override {
    std::discrete_distribution<int> disc_dist(operatorsweights_.begin(),
                                              operatorsweights_.end());
    std::uniform_real_distribution<double> distu;

    for (int i = 0; i < nv_; i++) {
      // pick a random operator in possible ones according to the provided
      // weights
      int op = disc_dist(this->GetRandomEngine());

      move_operators_.FindConn(op, v_, mel_, tochange_, newconfs_);

      double p = distu(this->GetRandomEngine());
      std::size_t exit_state = 0;
      double cumulative_prob = std::real(mel_[0]);
      while (p > cumulative_prob) {
        exit_state++;
        cumulative_prob += std::real(mel_[exit_state]);
      }

      auto exlog = std::exp(GetMachine().LogValDiff(v_, tochange_[exit_state],
                                            newconfs_[exit_state], lt_));
      double ratio = this->GetMachineFunc()(exlog);

      // Metropolis acceptance test
      if (ratio > distu(this->GetRandomEngine())) {
        accept_[0] += 1;
        GetMachine().UpdateLookup(v_, tochange_[exit_state], newconfs_[exit_state],
                          lt_);
        GetHilbert().UpdateConf(v_, tochange_[exit_state], newconfs_[exit_state]);
      }
      moves_[0] += 1;
    }
  }

  Eigen::VectorXd Visible() override { return v_; }

  void SetVisible(const Eigen::VectorXd &v) override { v_ = v; }

  AbstractMachine::VectorType DerLogVisible() override {
    return GetMachine().DerLog(v_, lt_);
  }

  Eigen::VectorXd Acceptance() const override {
    Eigen::VectorXd acc = accept_;
    for (int i = 0; i < 1; i++) {
      acc(i) /= moves_(i);
    }
    return acc;
  }

  static void CheckMoveOperators(const LocalOperator &move_operators) {
    if (move_operators.Size() == 0) {
      throw InvalidInputError("No valid MoveOperators provided");
    }

    const auto local_matrices = move_operators.LocalMatrices();
    const auto acting_on = move_operators.ActingOn();

    std::set<int> touched_sites;
    for (std::size_t c = 0; c < local_matrices.size(); c++) {
      // check if matrix of this operator is stochastic
      bool is_stochastic = true;
      bool is_complex = false;
      bool is_definite_positive = true;
      bool is_symmetric = true;
      bool is_offdiagonal = true;

      const double epsilon = 1.0e-6;
      double sum_diagonal = 0;
      for (std::size_t i = 0; i < local_matrices[c].size(); i++) {
        double sum_column = 0.;
        for (std::size_t j = 0; j < local_matrices[c].size(); j++) {
          if (std::abs(std::imag(local_matrices[c][i][j])) > epsilon) {
            is_complex = true;
            is_stochastic = false;
            break;
          }
          if (std::real(local_matrices[c][i][j]) < 0) {
            is_definite_positive = false;
            is_stochastic = false;
            break;
          }
          if (std::abs(local_matrices[c][i][j] - local_matrices[c][j][i]) >
              epsilon) {
            is_symmetric = false;
            is_stochastic = false;
            break;
          }
          sum_column += std::real(local_matrices[c][i][j]);
        }
        if (std::abs(sum_column - 1.) > epsilon) {
          is_stochastic = false;
        }
        sum_diagonal += std::real(local_matrices[c][i][i]);
      }
      is_offdiagonal =
          std::abs(sum_diagonal - local_matrices[c].size()) > epsilon;
      if (is_offdiagonal == false) {
        throw InvalidInputError("MoveOperators has a diagonal move operator");
      }
      if (is_complex == true) {
        throw InvalidInputError("MoveOperators has complex matrix elements");
      }
      if (is_definite_positive == false) {
        throw InvalidInputError("MoveOperators has negative matrix elements");
      }
      if (is_symmetric == false) {
        throw InvalidInputError("MoveOperators is not symmetric");
      }
      if (is_stochastic == false) {
        throw InvalidInputError("MoveOperators is not stochastic");
      }

      for (std::size_t i = 0; i < acting_on[c].size(); i++) {
        touched_sites.insert(acting_on[c][i]);
      }
    }

    if (static_cast<int>(touched_sites.size()) !=
        move_operators.GetHilbert().Size()) {
      InfoMessage() << "Warning: MoveOperators appear not to act on "
                       "all sites of the space:"
                    << std::endl;
      InfoMessage() << "Check ergodicity" << std::endl;
    }
  }
};
}  // namespace netket

#endif
