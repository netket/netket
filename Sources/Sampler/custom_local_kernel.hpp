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

#ifndef NETKET_CUSTOM_LOCAL_KERNEL_HPP
#define NETKET_CUSTOM_LOCAL_KERNEL_HPP

#include <Eigen/Core>
#include <set>
#include "Operator/local_operator.hpp"

#include "Utils/exceptions.hpp"
#include "Utils/messages.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Metropolis sampling using custom moves provided by user
class CustomLocalKernel {
  LocalOperator move_operators_;
  std::vector<double> operatorsweights_;

  // number of visible units
  const Index nv_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  Index nstates_;

  std::vector<double> localstates_;

  std::discrete_distribution<Index> disc_dist_;

  std::uniform_real_distribution<double> distu_;

 public:
  CustomLocalKernel(const AbstractMachine &psi,
                    const LocalOperator &move_operators,
                    const std::vector<double> &move_weights = {})
      : move_operators_(move_operators), nv_(psi.Nvisible()) {
    Init(psi, move_weights);
  }

  void Init(const AbstractMachine &psi,
            const std::vector<double> &move_weights) {
    CheckMoveOperators(move_operators_);

    nstates_ = psi.GetHilbert().LocalSize();
    localstates_ = psi.GetHilbert().LocalStates();

    if (nv_ != move_operators_.GetHilbert().Size() ||
        nstates_ != move_operators_.GetHilbert().LocalSize()) {
      throw InvalidInputError(
          "Move operators in CustomSampler act on a different hilbert space "
          "than the Machine");
    }

    if (!psi.GetHilbert().IsDiscrete()) {
      throw InvalidInputError(
          "Custom Metropolis sampler works only for discrete Hilbert spaces");
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

    disc_dist_ = std::discrete_distribution<Index>(operatorsweights_.begin(),
                                                   operatorsweights_.end());

    distu_ = std::uniform_real_distribution<double>{};
  }

  void operator()(Eigen::Ref<const RowMatrix<double>> v,
                  Eigen::Ref<RowMatrix<double>> vnew,
                  Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction) {
    vnew = v;

    for (int i = 0; i < v.rows(); i++) {
      // pick a random operator in possible ones according to the provided
      // weights
      Index op = disc_dist_(GetRandomEngine());

      move_operators_.FindConn(op, v.row(i), mel_, tochange_, newconfs_);

      double p = distu_(GetRandomEngine());
      Index exit_state = 0;
      double cumulative_prob = std::real(mel_[0]);
      while (p > cumulative_prob) {
        exit_state++;
        cumulative_prob += std::real(mel_[exit_state]);
      }

      move_operators_.GetHilbert().UpdateConf(
          vnew.row(i), tochange_[exit_state], newconfs_[exit_state]);
    }

    log_acceptance_correction.setZero();
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
