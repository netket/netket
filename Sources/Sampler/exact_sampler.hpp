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

#ifndef NETKET_EXACT_SAMPLER_HPP
#define NETKET_EXACT_SAMPLER_HPP

#include <Eigen/Core>
#include "Sampler/abstract_sampler.hpp"
#include "Utils/messages.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

// Exact sampling using heat bath, mostly for testing purposes on small systems
class ExactSampler : public AbstractSampler {
  // number of visible units
  const int nv_;

  // states of visible units
  Eigen::VectorXd v_;
  int state_index_;

  int mynode_;
  int totalnodes_;

  const HilbertIndex& hilbert_index_;

  const int dim_;

  std::discrete_distribution<int> dist_;

  std::vector<Complex> logpsivals_;
  std::vector<double> psivals_;

 public:
  explicit ExactSampler(AbstractMachine& psi)
      : AbstractSampler(psi),
        nv_(GetMachine().GetHilbert().Size()),
        hilbert_index_(GetMachine().GetHilbert().GetIndex()),
        dim_(hilbert_index_.NStates()) {
    Init();
  }

  void Init() {
    v_.resize(nv_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (!GetMachine().GetHilbert().IsDiscrete()) {
      throw InvalidInputError(
          "Exact sampler works only for discrete "
          "Hilbert spaces");
    }

    Reset(true);

    InfoMessage() << "Exact sampler is ready " << std::endl;
  }

  void Reset(bool initrandom) override {
    double logmax = -std::numeric_limits<double>::infinity();

    logpsivals_.resize(dim_);
    psivals_.resize(dim_);

    for (int i = 0; i < dim_; ++i) {
      auto v = hilbert_index_.NumberToState(i);
      logpsivals_[i] = GetMachine().LogValSingle(v);
      logmax = std::max(logmax, std::real(logpsivals_[i]));
    }

    for (int i = 0; i < dim_; ++i) {
      psivals_[i] = this->GetMachineFunc()(std::exp(logpsivals_[i] - logmax));
    }

    dist_ = std::discrete_distribution<int>(psivals_.begin(), psivals_.end());

    if (initrandom) {
      state_index_ = dist_(this->GetRandomEngine());
      v_ = hilbert_index_.NumberToState(state_index_);
    }
  }

  void Sweep() override {
    state_index_ = dist_(this->GetRandomEngine());
    v_ = hilbert_index_.NumberToState(state_index_);
  }

  std::pair<Eigen::Ref<const RowMatrix<double>>,
            Eigen::Ref<const Eigen::VectorXcd>>
  CurrentState() const override {
    return {v_.transpose(),
            Eigen::Map<const Eigen::VectorXcd>{&logpsivals_[state_index_], 1}};
  }

  void SetVisible(Eigen::Ref<const RowMatrix<double>> v) override {
    CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()},
               {1, GetMachine().Nvisible()});
    v_ = v.row(0);
    state_index_ = hilbert_index_.StateToNumber(v_);
  }

  double Acceptance() const noexcept { return 1; }

  Index BatchSize() const noexcept override { return 1; }

  void SetMachineFunc(MachineFunction machine_func) override {
    AbstractSampler::SetMachineFunc(machine_func);
    Reset(true);
  }
};

}  // namespace netket

#endif
