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
#include "../Hamiltonian/local_operator.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_sampler.hpp"

namespace netket {

// Metropolis sampling using custom moves provided by user
template <class WfType>
class CustomSampler : public AbstractSampler<WfType> {
  WfType &psi_;
  const AbstractHilbert &hilbert_;
  std::vector<LocalOperator> move_operators_;
  std::vector<double> operatorsweights_;

  // number of visible units
  const int nv_;

  netket::default_random_engine rgen_;

  // states of visible units
  Eigen::VectorXd v_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  // Look-up tables
  typename WfType::LookupType lt_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<std::complex<double>> mel_;

  int nstates_;
  std::vector<double> localstates_;

 public:
  using MatType = std::vector<std::vector<std::complex<double>>>;

  template <class Ptype>
  explicit CustomSampler(WfType &psi, const Ptype &pars)
      : psi_(psi), hilbert_(psi.GetHilbert()), nv_(hilbert_.Size()) {
    CheckFieldExists(pars, "MoveOperators");

    // TODO
    // if (!pars["MoveOperators"].is_array()) {
    //   throw InvalidInputError("MoveOperators is not an array");
    // }
    CheckFieldExists(pars, "ActingOn");

    // if (!pars["ActingOn"].is_array()) {
    //   throw InvalidInputError("ActingOn is not an array");
    // }

    std::vector<MatType> jop =
        FieldVal<std::vector<MatType>>(pars, "MoveOperators");
    std::vector<std::vector<int>> sites =
        FieldVal<std::vector<std::vector<int>>>(pars, "ActingOn");

    if (sites.size() != jop.size()) {
      throw InvalidInputError(
          "The custom sampler definition is inconsistent (between "
          "MoveOperators and ActingOn sizes); Check that ActingOn is defined");
    }

    std::set<int> touched_sites;
    for (std::size_t c = 0; c < jop.size(); c++) {
      // check if matrix of this operator is stochastic
      bool is_stochastic = true;
      bool is_complex = false;
      bool is_definite_positive = true;
      bool is_symmetric = true;
      const double epsilon = 1.0e-6;
      for (std::size_t i = 0; i < jop[c].size(); i++) {
        double sum_column = 0.;
        for (std::size_t j = 0; j < jop[c].size(); j++) {
          if (std::abs(jop[c][i][j].imag()) > epsilon) {
            is_complex = true;
            is_stochastic = false;
            break;
          }
          if (jop[c][i][j].real() < 0) {
            is_definite_positive = false;
            is_stochastic = false;
            break;
          }
          if (std::abs(jop[c][i][j].real() - jop[c][j][i].real()) > epsilon) {
            is_symmetric = false;
            is_stochastic = false;
            break;
          }
          sum_column += jop[c][i][j].real();
        }
        if (std::abs(sum_column - 1.) > epsilon) {
          is_stochastic = false;
        }
      }
      if (is_complex == true) {
        InfoMessage() << "Warning: MoveOperators " << c
                      << " has complex matrix elements" << std::endl;
      }
      if (is_definite_positive == false) {
        InfoMessage() << "Warning: MoveOperators " << c
                      << " has negative matrix elements" << std::endl;
      }
      if (is_symmetric == false) {
        InfoMessage() << "Warning: MoveOperators " << c << " is not symmetric"
                      << std::endl;
      }
      if (is_stochastic == false) {
        InfoMessage() << "Warning: MoveOperators " << c << " is not stochastic"
                      << std::endl;
        InfoMessage() << "MoveOperators " << c << " is discarded" << std::endl;
      }

      else {
        move_operators_.push_back(LocalOperator(hilbert_, jop[c], sites[c]));
        for (std::size_t i = 0; i < sites[c].size(); i++) {
          touched_sites.insert(sites[c][i]);
        }
      }
    }

    if (move_operators_.size() == 0) {
      throw InvalidInputError("No valid MoveOperators provided");
    }

    if (static_cast<int>(touched_sites.size()) != hilbert_.Size()) {
      InfoMessage() << "Warning: MoveOperators appear not to act on "
                       "all sites of the sample:"
                    << std::endl;
      InfoMessage() << "Check ergodicity" << std::endl;
    }

    if (FieldExists(pars, "MoveWeights")) {
      // if (!pars["MoveWeights"].is_array()) {
      //   throw InvalidInputError("MoveWeights is not an array");
      // }
      const std::vector<double> opw =
          FieldVal<std::vector<double>>(pars, "MoveWeights");
      operatorsweights_ = opw;

      if (operatorsweights_.size() != jop.size()) {
        throw InvalidInputError(
            "The custom sampler definition is inconsistent (between "
            "MoveWeights and MoveOperators sizes)");
      }

    } else {  // By default the stochastic operators are drawn uniformly
      operatorsweights_.resize(move_operators_.size(), 1.0);
    }

    Init();
  }

  void Init() {
    v_.resize(nv_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (!hilbert_.IsDiscrete()) {
      throw InvalidInputError(
          "Custom Metropolis sampler works only for discrete Hilbert spaces");
    }

    accept_.resize(1);
    moves_.resize(1);

    nstates_ = hilbert_.LocalSize();
    localstates_ = hilbert_.LocalStates();

    Seed();

    Reset(true);

    InfoMessage() << "Custom Metropolis sampler is ready " << std::endl;
  }

  void Seed(int baseseed = 0) {
    std::random_device rd;
    std::vector<int> seeds(totalnodes_);

    if (mynode_ == 0) {
      for (int i = 0; i < totalnodes_; i++) {
        seeds[i] = rd() + baseseed;
      }
    }

    SendToAll(seeds);

    rgen_.seed(seeds[mynode_]);
  }

  void Reset(bool initrandom = false) override {
    if (initrandom) {
      hilbert_.RandomVals(v_, rgen_);
    }

    psi_.InitLookup(v_, lt_);

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
      int op = disc_dist(rgen_);
      move_operators_[op].FindConn(v_, mel_, tochange_, newconfs_);

      double p = distu(rgen_);
      std::size_t exit_state = 0;
      double cumulative_prob = mel_[0].real();
      while (p > cumulative_prob) {
        exit_state++;
        cumulative_prob += mel_[exit_state].real();
      }

      double ratio = std::norm(std::exp(psi_.LogValDiff(
          v_, tochange_[exit_state], newconfs_[exit_state], lt_)));

      // Metropolis acceptance test
      if (ratio > distu(rgen_)) {
        accept_[0] += 1;
        psi_.UpdateLookup(v_, tochange_[exit_state], newconfs_[exit_state],
                          lt_);
        hilbert_.UpdateConf(v_, tochange_[exit_state], newconfs_[exit_state]);
      }
      moves_[0] += 1;
    }
  }

  Eigen::VectorXd Visible() override { return v_; }

  void SetVisible(const Eigen::VectorXd &v) override { v_ = v; }

  WfType &Psi() override { return psi_; }

  Eigen::VectorXd Acceptance() const override {
    Eigen::VectorXd acc = accept_;
    for (int i = 0; i < 1; i++) {
      acc(i) /= moves_(i);
    }
    return acc;
  }
};
}  // namespace netket

#endif
