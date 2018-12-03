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

#ifndef NETKET_SAMPLER_HPP
#define NETKET_SAMPLER_HPP

#include <memory>
#include <set>
#include "Graph/graph.hpp"
#include "Operator/operator.hpp"
#include "Utils/memory_utils.hpp"
#include "Utils/parallel_utils.hpp"
#include "abstract_sampler.hpp"
#include "custom_sampler.hpp"
#include "custom_sampler_pt.hpp"
#include "exact_sampler.hpp"
#include "metropolis_exchange.hpp"
#include "metropolis_exchange_pt.hpp"
#include "metropolis_hamiltonian.hpp"
#include "metropolis_hamiltonian_pt.hpp"
#include "metropolis_hop.hpp"
#include "metropolis_local.hpp"
#include "metropolis_local_pt.hpp"
#include "mpark/variant.hpp"

namespace netket {
template <class WfType>
class Sampler : public AbstractSampler<WfType> {
  using VariantType =
      mpark::variant<CustomSampler<WfType>, CustomSamplerPt<WfType>,
                     MetropolisLocal<WfType>, MetropolisLocalPt<WfType>,
                     MetropolisExchange<WfType>, MetropolisExchangePt<WfType>,
                     MetropolisHamiltonian<WfType>,
                     MetropolisHamiltonianPt<WfType>, MetropolisHop<WfType>,
                     ExactSampler<WfType>>;

 private:
  VariantType obj_;

 public:
  Sampler(VariantType obj) : obj_(obj) {}

  void Reset(bool initrandom = false) override {
    mpark::visit([initrandom](auto &&obj) { obj.Reset(initrandom); }, obj_);
  }

  void Sweep() override {
    mpark::visit([](auto &&obj) { obj.Sweep(); }, obj_);
  }

  Eigen::VectorXd Visible() override {
    return mpark::visit([](auto &&obj) { return obj.Visible(); }, obj_);
  }

  void SetVisible(const Eigen::VectorXd &v) override {
    mpark::visit([v](auto &&obj) { obj.SetVisible(v); }, obj_);
  }

  WfType GetMachine() override {
    return mpark::visit([](auto &&obj) { return obj.GetMachine(); }, obj_);
  }

  void SetMachineParameters(typename WfType::VectorConstRefType pars) override {
    return mpark::visit(
        [pars](auto &&obj) { return obj.SetMachineParameters(pars); }, obj_);
  }

  Eigen::VectorXd Acceptance() const override {
    return mpark::visit([](auto &&obj) { return obj.Acceptance(); }, obj_);
  }

  Hilbert GetHilbert() const override {
    return mpark::visit([](auto &&obj) { return obj.GetHilbert(); }, obj_);
  }
};
}  // namespace netket

#endif
