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

#include "abstract_hilbert.hh"
#include "Json/json.hh"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#ifndef NETKET_BOSONS_HH
#define NETKET_BOSONS_HH

namespace netket {

/**
  Hilbert space for integer or bosons.
  The hilbert space is truncated to some maximum occupation number.
*/

class Boson : public AbstractHilbert {

  int nsites_;

  std::vector<double> local_;

  // total number of bosons
  // if constraint is activated
  int nbosons_;

  bool constraintN_;

  // maximum local occupation number
  int nmax_;

  int nstates_;

public:
  explicit Boson(const json &pars) {

    if (!FieldExists(pars["Hilbert"], "Nsites")) {
      std::cerr << "Nsites is not defined" << std::endl;
    }

    nsites_ = pars["Hilbert"]["Nsites"];

    if (!FieldExists(pars["Hilbert"], "Nmax")) {
      std::cerr << "Nmax is not defined" << std::endl;
    }

    nmax_ = pars["Hilbert"]["Nmax"];

    Init();

    if (FieldExists(pars["Hilbert"], "Nbosons")) {
      SetNbosons(pars["Hilbert"]["Nbosons"]);
    } else {
      constraintN_ = false;
    }
  }

  void Init() {

    if (nsites_ <= 0) {
      std::cerr << "Invalid number of sites" << std::endl;
      std::abort();
    }

    if (nmax_ > nsites_) {
      nmax_ = nsites_;
    }
    if (nmax_ <= 0) {
      std::cerr << "Invalid maximum occupation number" << std::endl;
      std::abort();
    }

    nstates_ = nmax_ + 1;

    local_.resize(nstates_);

    for (int i = 0; i < nstates_; i++) {
      local_[i] = i;
    }
  }

  void SetNbosons(int nbosons) {
    constraintN_ = true;
    nbosons_ = nbosons;

    if (nbosons_ > nsites_ * nmax_) {
      std::cerr << "Cannot set the desired number of bosons" << std::endl;
      std::abort();
    }
  }

  bool IsDiscrete() const { return true; }

  int LocalSize() const { return nstates_; }

  int Size() const { return nsites_; }

  std::vector<double> LocalStates() const { return local_; }

  void RandomVals(Eigen::VectorXd &state,
                  netket::default_random_engine &rgen) const {

    assert(state.size() == nsites_);

    if (!constraintN_) {
      std::uniform_int_distribution<int> distribution(0, nstates_ - 1);
      // unconstrained random
      for (int i = 0; i < state.size(); i++) {
        state(i) = distribution(rgen);
      }
    } else {
      state.setZero();

      std::uniform_int_distribution<int> distribution(0, nsites_ - 1);
      for (int i = 0; i < nbosons_; i++) {
        int rsite = distribution(rgen);

        while (state(rsite) >= nmax_) {
          rsite = distribution(rgen);
        }

        state(rsite) += 1;
      }
    }
  }

  bool CheckConstraint(Eigen::VectorXd &v) const {

    int tot = 0;
    for (int i = 0; i < v.size(); i++) {
      tot += int(v(i));
    }

    return tot == nbosons_;
  }

  void UpdateConf(Eigen::VectorXd &v, const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const {

    assert(v.size() == nsites_);

    int i = 0;
    for (auto sf : tochange) {
      v(sf) = newconf[i];
      i++;
      assert(v(sf) <= nmax_);
    }

    if (constraintN_) {
      assert(CheckConstraint(v));
    }
  }
};

} // namespace netket
#endif
