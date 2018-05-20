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

#include "abstract_hilbert.hpp"
#include "Json/json_helper.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#ifndef NETKET_SPIN_HPP
#define NETKET_SPIN_HPP

namespace netket {

/**
  Hilbert space for integer or half-integer spins.
  Notice that here integer values are always used to represent the local quantum
  numbers, such that for example if total spin is S=3/2, the allowed quantum
  numbers are -3,-1,1,3, and if S=1 we have -2,0,2.
*/

class Spin : public AbstractHilbert {

  double S_;
  double totalS_;
  bool constraintSz_;

  std::vector<double> local_;

  int nstates_;

  int nspins_;

public:
  explicit Spin(const json &pars) {

    int nspins;
    double S;

    if (!FieldExists(pars["Hilbert"], "Nspins")) {
      std::cerr << "Nspins is not defined" << std::endl;
    }

    nspins = pars["Hilbert"]["Nspins"];

    if (!FieldExists(pars["Hilbert"], "S")) {
      std::cerr << "S is not defined" << std::endl;
    }

    S = pars["Hilbert"]["S"];

    Init(nspins, S);

    if (FieldExists(pars["Hilbert"], "TotalSz")) {
      SetConstraint(pars["Hilbert"]["TotalSz"]);
    } else {
      constraintSz_ = false;
    }
  }

  void Init(int nspins, double S) {
    S_ = S;
    nspins_ = nspins;

    if (S <= 0) {
      std::cerr << "Invalid spin value" << std::endl;
      std::abort();
    }

    if (std::floor(2. * S) != 2. * S) {
      std::cerr << "Spin value is hot integer or half integer" << std::endl;
      std::abort();
    }

    nstates_ = std::floor(2. * S) + 1;

    local_.resize(nstates_);

    int sp = -std::floor(2. * S);
    for (int i = 0; i < nstates_; i++) {
      local_[i] = sp;
      sp += 2;
    }
  }

  void SetConstraint(double totalS) {
    constraintSz_ = true;
    totalS_ = totalS;
  }

  bool IsDiscrete() const override { return true; }

  int LocalSize() const override { return nstates_; }

  int Size() const override { return nspins_; }

  std::vector<double> LocalStates() const override { return local_; }

  void RandomVals(Eigen::VectorXd &state,
                  netket::default_random_engine &rgen) const override {

    std::uniform_int_distribution<int> distribution(0, nstates_ - 1);

    assert(state.size() == nspins_);

    if (!constraintSz_) {
      // unconstrained random
      for (int i = 0; i < state.size(); i++) {
        state(i) = 2. * (distribution(rgen) - S_);
      }
    } else {

      if (S_ == 0.5) {
        int nup = nspins_ / 2 + int(totalS_);
        int ndown = nspins_ - nup;

        if ((nup - ndown) != int(2 * totalS_)) {
          std::cerr << "#Cannot fix the total magnetization " << std::endl;
          std::abort();
        }

        std::vector<double> vect(nspins_);

        for (int i = 0; i < nup; i++) {
          vect[i] = +1.;
        }
        for (int i = nup; i < nspins_; i++) {
          vect[i] = -1.;
        }

        // now random shuffle
        std::shuffle(vect.begin(), vect.end(), rgen);

        for (int i = 0; i < nspins_; i++) {
          state(i) = vect[i];
        }
        return;
      } else {
        std::vector<int> sites;
        for (int i = 0; i < nspins_; ++i)
          sites.push_back(i);

        state.setConstant(-2 * S_);
        int ss = nspins_;

        for (int i = 0; i < S_ * nspins_ + totalS_; ++i) {
          std::uniform_int_distribution<int> distribution_ss(0, ss - 1);
          int s = distribution_ss(rgen);
          state(sites[s]) += 2;
          if (state(sites[s]) > 2 * S_ - 1) {
            sites.erase(sites.begin() + s);
            ss -= 1;
          }
        }
      }
    }
  }

  void UpdateConf(Eigen::VectorXd &v, const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const override {

    assert(v.size() == nspins_);

    int i = 0;
    for (auto sf : tochange) {
      v(sf) = newconf[i];
      i++;
    }
  }
};

} // namespace netket
#endif
