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

#include "bosons.hpp"

namespace netket {

Boson::Boson(const AbstractGraph &graph, int const nmax)
    : graph_{graph}, nsites_{graph.Size()}, constraintN_{false}, nmax_{nmax} {
  Init();
}

Boson::Boson(const AbstractGraph &graph, int const nmax, int const nbosons)
    : graph_{graph}, nsites_{graph.Size()}, nmax_{nmax} {
  Init();
  SetNbosons(nbosons);
}

void Boson::Init() {
  assert(nsites_ > 0);

  if (nmax_ <= 0) {
    throw InvalidInputError("Invalid maximum occupation number");
  }

  nstates_ = nmax_ + 1;

  local_.resize(nstates_);

  for (int i = 0; i < nstates_; i++) {
    local_[i] = i;
  }
}

void Boson::SetNbosons(int const nbosons) {
  constraintN_ = true;
  nbosons_ = nbosons;

  if (nbosons_ > nsites_ * nmax_) {
    throw InvalidInputError("Cannot set the desired number of bosons");
  }
}

bool Boson::IsDiscrete() const { return true; }

int Boson::LocalSize() const { return nstates_; }

int Boson::Size() const { return nsites_; }

std::vector<double> Boson::LocalStates() const { return local_; }

const AbstractGraph &Boson::GetGraph() const noexcept { return graph_; }

void Boson::RandomVals(Eigen::Ref<Eigen::VectorXd> state,
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

bool Boson::CheckConstraint(Eigen::Ref<const Eigen::VectorXd> v) const {
  int tot = 0;
  for (int i = 0; i < v.size(); i++) {
    tot += std::round(v(i));
  }

  return tot == nbosons_;
}

void Boson::UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                       const std::vector<int> &tochange,
                       const std::vector<double> &newconf) const {
  assert(v.size() == nsites_);

  int i = 0;
  for (auto sf : tochange) {
    v(sf) = newconf[i];
    i++;
    assert(v(sf) <= nmax_);
    assert(v(sf) >= 0);
  }

  if (constraintN_) {
    assert(CheckConstraint(v));
  }
}

}  // namespace netket
