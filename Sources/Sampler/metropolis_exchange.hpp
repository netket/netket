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

#ifndef NETKET_METROPOLISEXCHANGE_HPP
#define NETKET_METROPOLISEXCHANGE_HPP

#include <Eigen/Core>
#include "Sampler/abstract_sampler.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Metropolis sampling generating local exchanges
class MetropolisExchange : public AbstractSampler {
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

  int sweep_size_;

  LogValAccumulator log_val_accumulator_;

 public:
  MetropolisExchange(const AbstractGraph &graph, AbstractMachine &psi,
                     int dmax = 1)
      : AbstractSampler(psi), nv_(GetMachine().GetHilbert().Size()) {
    Init(graph, dmax);
  }

  template <class G>
  void Init(const G &graph, int dmax) {
    v_.resize(nv_);

    GenerateClusters(graph, dmax);

    Reset(true);

    // Always use odd sweep size to avoid possible ergodicity problems
    if (nv_ % 2 == 0) {
      sweep_size_ = nv_ + 1;
    } else {
      sweep_size_ = nv_;
    }

    InfoMessage() << "Metropolis Exchange sampler is ready " << std::endl;
    InfoMessage() << dmax << " is the maximum distance for exchanges"
                  << std::endl;
  }

  template <class G>
  void GenerateClusters(const G &graph, int dmax) {
    auto dist = graph.AllDistances();

    assert(int(dist.size()) == nv_);

    for (int i = 0; i < nv_; i++) {
      for (int j = 0; j < nv_; j++) {
        if (dist[i][j] <= dmax && i != j) {
          clusters_.push_back({i, j});
        }
      }
    }
  }

  void Reset(bool initrandom = false) override {
    if (initrandom) {
      if (initrandom) {
        GetMachine().GetHilbert().RandomVals(v_, this->GetRandomEngine());
      }
    }

    lt_ = GetMachine().InitLookup(v_);
    log_val_accumulator_ = GetMachine().LogValSingle(v_, lt_);
    accept_ = 0;
    moves_ = 0;
  }

  void Sweep() override {
    std::vector<int> tochange(2);
    std::uniform_real_distribution<double> distu;
    std::uniform_int_distribution<int> distcl(0, clusters_.size() - 1);

    std::vector<double> newconf(2);

    for (int i = 0; i < sweep_size_; i++) {
      int rcl = distcl(this->GetRandomEngine());
      assert(rcl < int(clusters_.size()));
      int si = clusters_[rcl][0];
      int sj = clusters_[rcl][1];

      assert(si < nv_ && sj < nv_);

      if (std::abs(v_(si) - v_(sj)) > std::numeric_limits<double>::epsilon()) {
        tochange = clusters_[rcl];
        newconf[0] = v_(sj);
        newconf[1] = v_(si);

        const auto log_val_diff =
            GetMachine().LogValDiff(v_, tochange, newconf, lt_);
        auto explo = std::exp(log_val_diff);

        double ratio = NETKET_SAMPLER_APPLY_MACHINE_FUNC(explo);

        if (ratio > distu(this->GetRandomEngine())) {
          ++accept_;
          GetMachine().UpdateLookup(v_, tochange, newconf, lt_);
          GetMachine().GetHilbert().UpdateConf(v_, tochange, newconf);
          log_val_accumulator_ += log_val_diff;
        }
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

  Index BatchSize() const noexcept override { return 1; }

  NETKET_SAMPLER_SET_VISIBLE_DEFAULT(v_)
  NETKET_SAMPLER_ACCEPTANCE_DEFAULT(accept_, moves_)
};

}  // namespace netket

#endif
