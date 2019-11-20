// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#include "custom_local_kernel.hpp"

namespace netket {

CustomLocalKernel::CustomLocalKernel(const AbstractMachine &psi,
                                     const LocalOperator &move_operators,
                                     const std::vector<double> &move_weights)
    : move_operators_(move_operators) {
  Init(psi.GetHilbert(), move_weights);
  NETKET_CHECK(
      psi.Nvisible() == move_operators_.GetHilbert().Size() &&
          nstates_ == move_operators_.GetHilbert().LocalSize(),
      InvalidInputError,
      "Move operators in CustomSampler act on a different hilbert space "
      "than the Machine");
}

CustomLocalKernel::CustomLocalKernel(const LocalOperator &move_operators,
                                     const std::vector<double> &move_weights)
    : move_operators_(move_operators) {
  Init(move_operators_.GetHilbert(), move_weights);
}

void CustomLocalKernel::Init(const AbstractHilbert &hilb,
                             const std::vector<double> &move_weights) {
  CheckMoveOperators(move_operators_);

  nstates_ = hilb.LocalSize();
  localstates_ = hilb.LocalStates();

  NETKET_CHECK(
      hilb.IsDiscrete(), InvalidInputError,
      "Custom Metropolis sampler works only for discrete Hilbert spaces");

  if (move_weights.size()) {
    operatorsweights_ = move_weights;

    NETKET_CHECK(operatorsweights_.size() == move_operators_.Size(),
                 InvalidInputError,
                 "The custom sampler definition is inconsistent (between "
                 "MoveWeights and MoveOperators sizes)");

  } else {  // By default the stochastic operators are drawn uniformly
    operatorsweights_.resize(move_operators_.Size(), 1.0);
  }

  disc_dist_ = std::discrete_distribution<Index>(operatorsweights_.begin(),
                                                 operatorsweights_.end());
}

void CustomLocalKernel::operator()(
    Eigen::Ref<const RowMatrix<double>> v, Eigen::Ref<RowMatrix<double>> vnew,
    Eigen::Ref<Eigen::ArrayXd> log_acceptance_correction) {
  vnew = v;

  for (Index i{0}; i < v.rows(); i++) {
    // pick a random operator in possible ones according to the provided
    // weights
    Index op = disc_dist_(GetRandomEngine());

    move_operators_.FindConn(op, v.row(i), mel_, tochange_, newconfs_);

    double p = std::uniform_real_distribution<double>{}(GetRandomEngine());
    Index exit_state = 0;
    double cumulative_prob = std::real(mel_[0]);
    while (p > cumulative_prob) {
      exit_state++;
      cumulative_prob += std::real(mel_[exit_state]);
    }

    move_operators_.GetHilbert().UpdateConf(vnew.row(i), tochange_[exit_state],
                                            newconfs_[exit_state]);
  }

  log_acceptance_correction.setZero();
}

void CustomLocalKernel::CheckMoveOperators(
    const LocalOperator &move_operators) {
  NETKET_CHECK(move_operators.Size() != 0, InvalidInputError,
               "No valid MoveOperators provided");

  const auto local_matrices = move_operators.LocalMatrices();
  const auto acting_on = move_operators.ActingOn();

  std::set<int> touched_sites;
  for (std::size_t c = 0; c < local_matrices.size(); c++) {
    // check if matrix of this operator is stochastic
    bool is_stochastic = true;
    bool is_real = true;
    bool is_definite_positive = true;
    bool is_symmetric = true;
    bool is_offdiagonal = true;

    const double epsilon = 1.0e-6;
    double sum_diagonal = 0;
    for (std::size_t i = 0; i < local_matrices[c].size(); i++) {
      double sum_column = 0.;
      for (std::size_t j = 0; j < local_matrices[c].size(); j++) {
        if (std::abs(std::imag(local_matrices[c][i][j])) > epsilon) {
          is_real = false;
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

    NETKET_CHECK(is_offdiagonal, InvalidInputError,
                 "MoveOperators has a diagonal move operator");

    NETKET_CHECK(is_real, InvalidInputError,
                 "MoveOperators has complex matrix elements");

    NETKET_CHECK(is_definite_positive, InvalidInputError,
                 "MoveOperators has negative matrix elements");

    NETKET_CHECK(is_symmetric, InvalidInputError,
                 "MoveOperators is not symmetric");

    NETKET_CHECK(is_stochastic, InvalidInputError,
                 "MoveOperators is not stochastic");

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

}  // namespace netket
