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

#ifndef NETKET_LOCAL_OPERATOR_HPP
#define NETKET_LOCAL_OPERATOR_HPP

#include <mpi.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <vector>
#include "Hilbert/abstract_hilbert.hpp"
#include "Utils/kronecker_product.hpp"
#include "Utils/next_variation.hpp"
#include "abstract_operator.hpp"

namespace netket {

/**
    Class for local operators acting on a list of sites and for generic local
    Hilbert spaces.
*/

class LocalOperator : public AbstractOperator {
 public:
  using MelType = std::complex<double>;
  using MatType = std::vector<std::vector<MelType>>;
  using SiteType = std::vector<int>;
  using MapType = std::map<std::vector<double>, int>;
  using StateType = std::vector<std::vector<double>>;
  using ConnType = std::vector<std::vector<int>>;
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

 private:
  std::shared_ptr<const AbstractHilbert> hilbert_;
  std::vector<MatType> mat_;
  std::vector<SiteType> sites_;

  std::vector<MapType> invstate_;
  std::vector<StateType> states_;
  std::vector<ConnType> connected_;

  std::size_t nops_;

 public:
  explicit LocalOperator(std::shared_ptr<const AbstractHilbert> hilbert,
                         const std::vector<MatType> &mat,
                         const std::vector<SiteType> &sites)
      : hilbert_(hilbert) {
    for (std::size_t i = 0; i < mat.size(); i++) {
      Push(mat[i], sites[i]);
    }
    Init();
  }

  explicit LocalOperator(std::shared_ptr<const AbstractHilbert> hilbert,
                         const MatType &mat, const SiteType &sites)
      : hilbert_(hilbert) {
    Push(mat, sites);
    // TODO sort sites and swap columns of mat accordingly
    Init();
  }

  void Push(const MatType &mat, const SiteType &sites) {
    std::size_t found = std::distance(
        sites_.begin(), std::find(sites_.begin(), sites_.end(), sites));
    if (found < sites_.size()) {
      for (std::size_t i = 0; i < mat_[found].size(); i++) {
        for (std::size_t j = 0; j < mat_[found][i].size(); j++) {
          mat_[found][i][j] += mat[i][j];
        }
      }
    } else {
      mat_.push_back(mat);
      sites_.push_back(sites);
    }
  }

  void Init() {
    if (!hilbert_->IsDiscrete()) {
      throw InvalidInputError(
          "Cannot construct operators on infinite local hilbert spaces");
    }
    if (sites_.size() != mat_.size()) {
      throw InvalidInputError(
          "Operators must specify a consistent set of acting_on specifiers");
    }

    nops_ = mat_.size();

    connected_.resize(nops_);
    states_.resize(nops_);
    invstate_.clear();
    invstate_.resize(nops_);

    for (std::size_t op = 0; op < nops_; op++) {
      const auto sites = sites_[op];
      const auto mat = mat_[op];
      auto &connected = connected_[op];
      auto &states = states_[op];
      auto &invstate = invstate_[op];

      if (*std::max_element(sites.begin(), sites.end()) >= hilbert_->Size() ||
          *std::min_element(sites.begin(), sites.end()) < 0) {
        throw InvalidInputError("Operator acts on an invalid set of sites");
      }

      auto localstates = hilbert_->LocalStates();
      const auto localsize = localstates.size();

      // Finding the non-zero matrix elements
      const double epsilon = 1.0e-6;

      connected.resize(mat.size());

      if (mat.size() != std::pow(localsize, sites.size())) {
        throw InvalidInputError(
            "Matrix size in operator is inconsistent with Hilbert space");
      }

      for (std::size_t i = 0; i < mat.size(); i++) {
        for (std::size_t j = 0; j < mat[i].size(); j++) {
          if (mat.size() != mat[i].size()) {
            throw InvalidInputError(
                "Matrix size in operator is inconsistent with Hilbert space");
          }

          if (i != j && std::abs(mat[i][j]) > epsilon) {
            connected[i].push_back(j);
          }
        }
      }

      // Construct the mapping
      // Internal index -> State
      std::vector<double> st(sites.size(), 0);

      do {
        states.push_back(st);
      } while (netket::next_variation(st.begin(), st.end(), localsize - 1));

      for (std::size_t i = 0; i < states.size(); i++) {
        for (std::size_t k = 0; k < states[i].size(); k++) {
          states[i][k] = localstates[states[i][k]];
        }
      }

      // Now construct the inverse mapping
      // State -> Internal index
      std::size_t k = 0;
      for (auto state : states) {
        invstate[state] = k;
        k++;
      }

      assert(k == mat.size());
    }
  }

  void FindConn(VectorConstRefType v, std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    assert(v.size() == hilbert_->Size());

    connectors.clear();
    newconfs.clear();
    mel.resize(0);

    AddConn(v, mel, connectors, newconfs);
  }

  void AddConn(VectorConstRefType v, std::vector<std::complex<double>> &mel,
               std::vector<std::vector<int>> &connectors,
               std::vector<std::vector<double>> &newconfs) const {
    if (mel.size() == 0) {
      connectors.resize(1);
      newconfs.resize(1);
      mel.resize(1);

      mel[0] = 0;
      connectors[0].resize(0);
      newconfs[0].resize(0);
    }

    for (std::size_t opn = 0; opn < nops_; opn++) {
      int st1 = StateNumber(v, opn);
      assert(st1 < int(mat_[opn].size()));
      assert(st1 < int(connected_[opn].size()));

      mel[0] += (mat_[opn][st1][st1]);

      // off-diagonal part
      for (auto st2 : connected_[opn][st1]) {
        connectors.push_back(sites_[opn]);
        assert(st2 < int(states_[opn].size()));
        newconfs.push_back(states_[opn][st2]);
        mel.push_back(mat_[opn][st1][st2]);
      }
    }
  }

  // FindConn for a specific operator
  void FindConn(std::size_t opn, VectorConstRefType v,
                std::vector<std::complex<double>> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const {
    assert(opn < mat_.size() && opn >= 0);

    mel.resize(1, 0.);
    connectors.resize(1);
    newconfs.resize(1);

    int st1 = StateNumber(v, opn);
    assert(st1 < int(mat_[opn].size()));
    assert(st1 < int(connected_[opn].size()));

    mel[0] = (mat_[opn][st1][st1]);

    // off-diagonal part
    for (auto st2 : connected_[opn][st1]) {
      connectors.push_back(sites_[opn]);
      assert(st2 < int(states_[opn].size()));
      newconfs.push_back(states_[opn][st2]);
      mel.push_back(mat_[opn][st1][st2]);
    }
  }

  inline int StateNumber(VectorConstRefType v, int opn) const {
    // TODO use a mask instead of copies
    std::vector<double> state(sites_[opn].size());
    for (std::size_t i = 0; i < sites_[opn].size(); i++) {
      state[i] = v(sites_[opn][i]);
    }
    return invstate_[opn].at(state);
  }

  // Product of two local operators, performing KroneckerProducts as necessary
  friend LocalOperator operator*(const LocalOperator &lhs,
                                 const LocalOperator &rhs) {
    // TODO
    // assert(lhs.Hilbert() == rhs.Hilbert());
    // check if sites have intersections, in that case this algorithm is wrong
    std::vector<MatType> mat;
    std::vector<SiteType> sites;

    for (std::size_t opn = 0; opn < lhs.mat_.size(); opn++) {
      for (std::size_t opn1 = 0; opn1 < rhs.mat_.size(); opn1++) {
        if (lhs.sites_[opn] == rhs.sites_[opn1]) {
          mat.push_back(netket::MatrixProduct(lhs.mat_[opn], rhs.mat_[opn1]));
          sites.push_back(lhs.sites_[opn]);
        } else {
          mat.push_back(
              netket::KroneckerProduct(lhs.mat_[opn], rhs.mat_[opn1]));
          SiteType sitesum = lhs.sites_[opn];
          sitesum.insert(sitesum.end(), rhs.sites_[opn1].begin(),
                         rhs.sites_[opn1].end());
          sites.push_back(sitesum);
        }
      }
    }
    return LocalOperator(lhs.GetHilbert(), mat, sites);
  }

  friend LocalOperator operator+(const LocalOperator &lhs,
                                 const LocalOperator &rhs) {
    assert(rhs.hilbert_->LocalStates().size() ==
           lhs.hilbert_->LocalStates().size());

    auto sites = lhs.sites_;
    auto mat = lhs.mat_;

    sites.insert(sites.end(), rhs.sites_.begin(), rhs.sites_.end());
    mat.insert(mat.end(), rhs.mat_.begin(), rhs.mat_.end());

    return LocalOperator(lhs.GetHilbert(), mat, sites);
  }

  template <class T>
  friend LocalOperator operator*(T lhs, const LocalOperator &rhs) {
    assert(std::imag(lhs) == 0.);
    auto mat = rhs.mat_;
    auto sites = rhs.sites_;

    for (std::size_t opn = 0; opn < mat.size(); opn++) {
      for (std::size_t i = 0; i < mat[opn].size(); i++) {
        for (std::size_t j = 0; j < mat[opn][i].size(); j++)
          mat[opn][i][j] *= lhs;
      }
    }

    return LocalOperator(rhs.GetHilbert(), mat, sites);
  }

  std::vector<MatType> LocalMatrices() const { return mat_; }
  std::vector<SiteType> ActingOn() const { return sites_; }

  std::shared_ptr<const AbstractHilbert> GetHilbert() const override {
    return hilbert_;
  }

  std::size_t Size() const { return mat_.size(); }
};  // namespace netket

}  // namespace netket
#endif
