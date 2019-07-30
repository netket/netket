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

#ifndef NETKET_METROPOLISHAMILTONIAN_HPP
#define NETKET_METROPOLISHAMILTONIAN_HPP

#include <Eigen/Core>
#include "Sampler/abstract_sampler.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Metropolis sampling generating transitions using the Hamiltonian
template <class H>
class MetropolisHamiltonian : public AbstractSampler {
  H &hamiltonian_;

  // number of visible units
  const int nv_;

  // states of visible units
  Eigen::VectorXd v_;

  Index accept_;
  Index moves_;

  // Look-up tables
  any lt_;

  std::vector<std::vector<int>> tochange_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<Complex> mel_;

  std::vector<std::vector<int>> tochange1_;
  std::vector<std::vector<double>> newconfs1_;
  std::vector<Complex> mel1_;

  Eigen::VectorXd v1_;

  int sweep_size_;

  LogValAccumulator log_val_accumulator_;

 public:
  MetropolisHamiltonian(AbstractMachine &psi, H &hamiltonian)
      : AbstractSampler(psi),
        hamiltonian_(hamiltonian),
        nv_(GetMachine().GetHilbert().Size()) {
    Init();
  }

  void Init() {
    v_.resize(nv_);

    if (!GetMachine().GetHilbert().IsDiscrete()) {
      throw InvalidInputError(
          "Hamiltonian Metropolis sampler works only for discrete "
          "Hilbert spaces");
    }

    Reset(true);

    // Always use odd sweep size to avoid possible ergodicity problems
    if (nv_ % 2 == 0) {
      sweep_size_ = nv_ + 1;
    } else {
      sweep_size_ = nv_;
    }

    InfoMessage() << "Hamiltonian Metropolis sampler is ready " << std::endl;
  }

  void Reset(bool initrandom = false) override {
    if (initrandom) {
      GetMachine().GetHilbert().RandomVals(v_, this->GetRandomEngine());
    }

    lt_ = GetMachine().InitLookup(v_);
    log_val_accumulator_ = GetMachine().LogValSingle(v_, lt_);
    accept_ = 0;
    moves_ = 0;
  }

  void Sweep() override {
    for (int i = 0; i < sweep_size_; i++) {
      hamiltonian_.FindConn(v_, mel_, tochange_, newconfs_);

      const double w1 = tochange_.size();

      std::uniform_int_distribution<int> distrs(0, tochange_.size() - 1);
      std::uniform_real_distribution<double> distu;

      // picking a random state to transit to
      int si = distrs(this->GetRandomEngine());

      // Inverse transition
      v1_ = v_;
      GetMachine().GetHilbert().UpdateConf(v1_, tochange_[si], newconfs_[si]);

      hamiltonian_.FindConn(v1_, mel1_, tochange1_, newconfs1_);

      double w2 = tochange1_.size();

      const auto lvd =
          GetMachine().LogValDiff(v_, tochange_[si], newconfs_[si], lt_);
      double ratio = NETKET_SAMPLER_APPLY_MACHINE_FUNC(std::exp(lvd)) * w1 / w2;

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

      // Metropolis acceptance test
      if (ratio > distu(this->GetRandomEngine())) {
        ++accept_;
        GetMachine().UpdateLookup(v_, tochange_[si], newconfs_[si], lt_);
        v_ = v1_;
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
