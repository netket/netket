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

#ifndef NETKET_METROPOLISHOP_HPP
#define NETKET_METROPOLISHOP_HPP

#include <Eigen/Core>
#include "Sampler/abstract_sampler.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Metropolis sampling generating local hoppings
class MetropolisHop : public AbstractSampler {
  // number of visible units
  const int nv_;

  // states of visible units
  Eigen::VectorXd v_;

  Index accept_;
  Index moves_;

  // clusters to do updates
  std::vector<std::vector<int>> clusters_;

  // Look-up tables
  any lt_;

  int nstates_;
  std::vector<double> localstates_;

  int sweep_size_;

  LogValAccumulator log_val_accumulator_;

 public:
  MetropolisHop(AbstractMachine &psi, int dmax = 1)
      : AbstractSampler(psi), nv_(GetMachine().GetHilbert().Size()) {
    Init(GetMachine().GetHilbert().GetGraph(), dmax);
  }

  void Init(const AbstractGraph &graph, int dmax) {
    v_.resize(nv_);

    nstates_ = GetMachine().GetHilbert().LocalSize();
    localstates_ = GetMachine().GetHilbert().LocalStates();

    GenerateClusters(graph, dmax);

    Reset(true);

    // Always use odd sweep size to avoid possible ergodicity problems
    if (nv_ % 2 == 0) {
      sweep_size_ = nv_ + 1;
    } else {
      sweep_size_ = nv_;
    }

    InfoMessage() << "Metropolis sampler is ready " << std::endl;
    InfoMessage() << "" << dmax
                  << " is the maximum distance for two-site clusters"
                  << std::endl;
  }

  void GenerateClusters(const AbstractGraph &graph, int dmax) {
    auto dist = graph.AllDistances();

    assert(int(dist.size()) == nv_);

    for (int i = 0; i < nv_; i++) {
      for (int j = 0; j < nv_; j++) {
        if (dist[i][j] <= dmax && i != j) {
          clusters_.push_back({i, j});
          clusters_.push_back({j, i});
        }
      }
    }
  }

  void Reset(bool initrandom = false) override {
    if (initrandom) {
      GetMachine().GetHilbert().RandomVals(v_);
    }

    lt_ = GetMachine().InitLookup(v_);
    log_val_accumulator_ = GetMachine().LogValSingle(v_, lt_);

    accept_ = 0;
    moves_ = 0;
  }

  void Sweep() override {
    std::vector<int> tochange(2);
    std::vector<double> newconf(2);
    std::vector<int> newstates(2);

    std::uniform_real_distribution<double> distu;
    std::uniform_int_distribution<int> distcl(0, clusters_.size() - 1);
    std::uniform_int_distribution<int> diststate(0, nstates_ - 1);

    for (int i = 0; i < sweep_size_; i++) {
      int rcl = distcl(GetRandomEngine());
      assert(rcl < int(clusters_.size()));
      int si = clusters_[rcl][0];
      int sj = clusters_[rcl][1];

      assert(si < nv_ && sj < nv_);

      tochange = clusters_[rcl];

      // picking a random state
      for (int k = 0; k < 2; k++) {
        newstates[k] = diststate(GetRandomEngine());
        newconf[k] = localstates_[newstates[k]];
      }

      // make sure that the new state is not equal to the current one
      while (std::abs(newconf[0] - v_(si)) <
                 std::numeric_limits<double>::epsilon() &&
             std::abs(newconf[1] - v_(sj)) <
                 std::numeric_limits<double>::epsilon()) {
        for (int k = 0; k < 2; k++) {
          newstates[k] = diststate(GetRandomEngine());
          newconf[k] = localstates_[newstates[k]];
        }
      }

      const auto lvd = GetMachine().LogValDiff(v_, tochange, newconf, lt_);
      double ratio = NETKET_SAMPLER_APPLY_MACHINE_FUNC(std::exp(lvd));

#ifndef NDEBUG
      const auto psival1 = GetMachine().LogValSingle(v_);
      if (std::abs(std::exp(GetMachine().LogValSingle(v_) -
                            GetMachine().LogValSingle(v_, lt_)) -
                   1.) > 1.0e-8) {
        std::cerr << GetMachine().LogValSingle(v_) << "  and LogVal with Lt is "
                  << GetMachine().LogValSingle(v_, lt_) << std::endl;
        std::abort();
      }
#endif

      if (ratio > distu(GetRandomEngine())) {
        ++accept_;
        GetMachine().UpdateLookup(v_, tochange, newconf, lt_);
        GetMachine().GetHilbert().UpdateConf(v_, tochange, newconf);
        log_val_accumulator_ += lvd;

#ifndef NDEBUG
        const auto psival2 = GetMachine().LogValSingle(v_);
        if (std::abs(std::exp(psival2 - psival1 - lvd) - 1.) > 1.0e-8) {
          std::cerr << psival2 - psival1 << " and logvaldiff is " << lvd
                    << std::endl;
          std::cerr << psival2 << " and LogVal with Lt is "
                    << GetMachine().LogValSingle(v_, lt_) << std::endl;
          std::abort();
        }
#endif
      }
      ++moves_;
    }
  }

  std::pair<Eigen::Ref<const RowMatrix<double>>,
            Eigen::Ref<const Eigen::VectorXcd>>
  CurrentState() const override {
    return {v_.transpose(), Eigen::Map<const Eigen::VectorXcd>{
                                &log_val_accumulator_.LogVal(), 1}};
  }

  NETKET_SAMPLER_SET_VISIBLE_DEFAULT(v_)
  NETKET_SAMPLER_ACCEPTANCE_DEFAULT(accept_, moves_)

  Index BatchSize() const noexcept override { return 1; }
};

}  // namespace netket

#endif
