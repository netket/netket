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

#ifndef NETKET_CUSTOMSAMPLERPT_HPP
#define NETKET_CUSTOMSAMPLERPT_HPP

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
class CustomSamplerPt : public AbstractSampler<WfType> {
  WfType &psi_;
  const AbstractHilbert &hilbert_;
  std::vector<LocalOperator> move_operators_;
  std::vector<double> operatorsweights_;
  // number of visible units
  const int nv_;

  netket::default_random_engine rgen_;

  // states of visible units
  // for each sampled temperature
  std::vector<Eigen::VectorXd> v_;

  Eigen::VectorXd accept_;
  Eigen::VectorXd moves_;

  int mynode_;
  int totalnodes_;

  // Look-up tables
  std::vector<typename WfType::LookupType> lt_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<std::complex<double>> mel_;

  int nstates_;
  std::vector<double> localstates_;

  int nrep_;
  std::vector<double> beta_;

 public:
  using MatType = std::vector<std::vector<std::complex<double>>>;

  explicit CustomSamplerPt(
      WfType &psi, const std::vector<MatType> &move_operators,
      const std::vector<std::vector<int>> &acting_on,
      std::vector<double> move_weights = std::vector<double>(),
      int nreplicas = 1)
      : psi_(psi),
        hilbert_(psi.GetHilbert()),
        nv_(hilbert_.Size()),
        nrep_(nreplicas) {
    if (acting_on.size() != move_operators.size()) {
      throw InvalidInputError(
          "The custom sampler definition is inconsistent (between "
          "MoveOperators and ActingOn sizes); Check that ActingOn is defined");
    }

    std::set<int> touched_sites;
    for (std::size_t c = 0; c < move_operators.size(); c++) {
      // check if matrix of this operator is stochastic
      bool is_stochastic = true;
      bool is_complex = false;
      bool is_definite_positive = true;
      bool is_symmetric = true;
      const double epsilon = 1.0e-6;
      for (std::size_t i = 0; i < move_operators[c].size(); i++) {
        double sum_column = 0.;
        for (std::size_t j = 0; j < move_operators[c].size(); j++) {
          if (std::abs(move_operators[c][i][j].imag()) > epsilon) {
            is_complex = true;
            is_stochastic = false;
            break;
          }
          if (move_operators[c][i][j].real() < 0) {
            is_definite_positive = false;
            is_stochastic = false;
            break;
          }
          if (std::abs(move_operators[c][i][j].real() -
                       move_operators[c][j][i].real()) > epsilon) {
            is_symmetric = false;
            is_stochastic = false;
            break;
          }
          sum_column += move_operators[c][i][j].real();
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
        move_operators_.push_back(
            LocalOperator(hilbert_, move_operators[c], acting_on[c]));
        for (std::size_t i = 0; i < acting_on[c].size(); i++) {
          touched_sites.insert(acting_on[c][i]);
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

    if (move_weights.size()) {
      operatorsweights_ = move_weights;

      if (operatorsweights_.size() != move_operators.size()) {
        throw InvalidInputError(
            "The custom sampler definition is inconsistent (between "
            "MoveWeights and MoveOperators sizes)");
      }

    } else {  // By default the stochastic operators are drawn uniformly
      operatorsweights_.resize(move_operators_.size(), 1.0);
    }

    Init();
  }

  // TODO remove
  template <class Ptype>
  explicit CustomSamplerPt(WfType &psi, const Ptype &pars)
      : psi_(psi), hilbert_(psi.GetHilbert()), nv_(hilbert_.Size()) {
    CheckFieldExists(pars, "Nreplicas");
    nrep_ = FieldVal<int>(pars, "Nreplicas");

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
    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (!hilbert_.IsDiscrete()) {
      throw InvalidInputError(
          "Custom Metropolis sampler works only for discrete Hilbert spaces");
    }

    nstates_ = hilbert_.LocalSize();
    localstates_ = hilbert_.LocalStates();

    v_.resize(nrep_);
    for (int i = 0; i < nrep_; i++) {
      v_[i].resize(nv_);
    }

    for (int i = 0; i < nrep_; i++) {
      beta_.push_back(1. - double(i) / double(nrep_));
    }

    lt_.resize(nrep_);

    accept_.resize(2 * nrep_);
    moves_.resize(2 * nrep_);

    Seed();

    Reset(true);

    InfoMessage() << "Custom Metropolis sampler with parallel tempering "
                     "is ready "
                  << std::endl;
    InfoMessage() << nrep_ << " replicas are being used" << std::endl;
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
      for (int i = 0; i < nrep_; i++) {
        hilbert_.RandomVals(v_[i], rgen_);
      }
    }

    for (int i = 0; i < nrep_; i++) {
      psi_.InitLookup(v_[i], lt_[i]);
    }

    accept_ = Eigen::VectorXd::Zero(2 * nrep_);
    moves_ = Eigen::VectorXd::Zero(2 * nrep_);
  }

  void LocalSweep(int rep) {
    std::discrete_distribution<int> disc_dist(operatorsweights_.begin(),
                                              operatorsweights_.end());
    std::uniform_real_distribution<double> distu;
    for (int i = 0; i < nv_; i++) {
      // pick a random operator in possible ones according to the provided
      // weights
      int op = disc_dist(rgen_);
      move_operators_[op].FindConn(v_[rep], mel_, tochange_, newconfs_);

      double p = distu(rgen_);
      std::size_t exit_state = 0;
      double cumulative_prob = mel_[0].real();
      while (p > cumulative_prob) {
        exit_state++;
        cumulative_prob += mel_[exit_state].real();
      }

      double ratio = std::norm(std::exp(
          beta_[rep] * psi_.LogValDiff(v_[rep], tochange_[exit_state],
                                       newconfs_[exit_state], lt_[rep])));

      // Metropolis acceptance test
      if (ratio > distu(rgen_)) {
        accept_(rep) += 1;
        psi_.UpdateLookup(v_[rep], tochange_[exit_state], newconfs_[exit_state],
                          lt_[rep]);
        hilbert_.UpdateConf(v_[rep], tochange_[exit_state],
                            newconfs_[exit_state]);
      }
      moves_(rep) += 1;
    }
  }

  void Sweep() override {
    // First we do local sweeps
    for (int i = 0; i < nrep_; i++) {
      LocalSweep(i);
    }

    // Temperature exchanges
    std::uniform_real_distribution<double> distribution(0, 1);

    for (int r = 1; r < nrep_; r += 2) {
      if (ExchangeProb(r, r - 1) > distribution(rgen_)) {
        Exchange(r, r - 1);
        accept_(nrep_ + r) += 1.;
        accept_(nrep_ + r - 1) += 1;
      }
      moves_(nrep_ + r) += 1.;
      moves_(nrep_ + r - 1) += 1;
    }

    for (int r = 2; r < nrep_; r += 2) {
      if (ExchangeProb(r, r - 1) > distribution(rgen_)) {
        Exchange(r, r - 1);
        accept_(nrep_ + r) += 1.;
        accept_(nrep_ + r - 1) += 1;
      }
      moves_(nrep_ + r) += 1.;
      moves_(nrep_ + r - 1) += 1;
    }
  }

  // computes the probability to exchange two replicas
  double ExchangeProb(int r1, int r2) {
    const double lf1 = 2 * std::real(psi_.LogVal(v_[r1], lt_[r1]));
    const double lf2 = 2 * std::real(psi_.LogVal(v_[r2], lt_[r2]));

    return std::exp((beta_[r1] - beta_[r2]) * (lf2 - lf1));
  }

  void Exchange(int r1, int r2) {
    std::swap(v_[r1], v_[r2]);
    std::swap(lt_[r1], lt_[r2]);
  }

  Eigen::VectorXd Visible() override { return v_[0]; }

  void SetVisible(const Eigen::VectorXd &v) override { v_[0] = v; }

  WfType &Psi() override { return psi_; }

  Eigen::VectorXd Acceptance() const override {
    Eigen::VectorXd acc = accept_;
    for (int i = 0; i < acc.size(); i++) {
      acc(i) /= moves_(i);
    }
    return acc;
  }
};
}  // namespace netket

#endif
