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

#ifndef NETKET_PAULIOPERATOR_HPP
#define NETKET_PAULIOPERATOR_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <vector>
#include "Graph/graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"
#include "abstract_operator.hpp"

namespace netket {

// Heisenberg model on an arbitrary graph

class PauliOperator : public AbstractOperator {
  const int nqubits_;
  const int noperators_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<std::complex<double>>> weights_;

  std::vector<std::vector<std::vector<int>>> zcheck_;

  const std::complex<double> I_;

  double cutoff_;

 public:
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

  explicit PauliOperator(std::shared_ptr<const AbstractHilbert> hilbert,
                         const std::vector<std::string> &ops,
                         const std::vector<std::complex<double>> &opweights,
                         double cutoff = 1e-10)
      : AbstractOperator(hilbert),
        nqubits_(hilbert->Size()),
        noperators_(ops.size()),
        I_(std::complex<double>(0, 1)),
        cutoff_(cutoff) {
    std::vector<std::vector<int>> tochange(noperators_);
    std::vector<std::complex<double>> weights = opweights;
    std::vector<std::vector<int>> zcheck(noperators_);
    int nchanges = 0;

    for (int i = 0; i < noperators_; i++) {
      if (ops[i].size() != std::size_t(nqubits_)) {
        throw InvalidInputError(
            "Operator size is inconsistent with number of qubits");
      }
      for (int j = 0; j < nqubits_; j++) {
        if (ops[i][j] == 'X') {
          tochange[i].push_back(j);
          nchanges++;
        }
        if (ops[i][j] == 'Y') {
          tochange[i].push_back(j);
          weights[i] *= I_;
          zcheck[i].push_back(j);
          nchanges++;
        }
        if (ops[i][j] == 'Z') {
          zcheck[i].push_back(j);
        }
      }
    }

    for (int i = 0; i < noperators_; i++) {
      auto tc = tochange[i];
      auto it = std::find(std::begin(tochange_), std::end(tochange_), tc);
      if (it != tochange_.end()) {
        int index = std::distance(tochange_.begin(), it);
        weights_[index].push_back(weights[i]);
        zcheck_[index].push_back(zcheck[i]);
      } else {
        tochange_.push_back(tc);
        weights_.push_back({weights[i]});
        zcheck_.push_back({zcheck[i]});
      }
    }

    InfoMessage() << "Pauli Operator created " << std::endl;
    InfoMessage() << "Nqubits = " << nqubits_ << std::endl;
    InfoMessage() << "Noperators = " << noperators_ << std::endl;
  }

  void FindConn(VectorConstRefType v, std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    assert(v.size() == nqubits_);

    connectors.resize(0);
    newconfs.clear();
    newconfs.resize(0);
    mel.clear();
    mel.resize(0);
    for (std::size_t i = 0; i < tochange_.size(); i++) {
      std::complex<double> mel_temp = 0.0;
      for (std::size_t j = 0; j < weights_[i].size(); j++) {
        std::complex<double> m_temp = weights_[i][j];
        for (auto k : zcheck_[i][j]) {
          assert(k >= 0 && k < v.size());
          if (int(std::round(v(k))) == 1) {
            m_temp *= -1.;
          }
        }
        mel_temp += m_temp;
      }
      if (std::abs(mel_temp) > cutoff_) {
        std::vector<double> newconf_temp(tochange_[i].size());
        int jj = 0;
        for (auto sj : tochange_[i]) {
          assert(sj < v.size() && sj >= 0);
          if (int(std::round(v(sj))) == 0) {
            newconf_temp[jj] = 1;
          } else {
            newconf_temp[jj] = 0;
          }
          jj++;
        }

        newconfs.push_back(newconf_temp);
        connectors.push_back(tochange_[i]);
        mel.push_back(mel_temp);
      }
    }
  }
};

}  // namespace netket

#endif
