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

#ifndef NETKET_CUSTOM_HILBERT_HPP
#define NETKET_CUSTOM_HILBERT_HPP

#include "Graph/abstract_graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"

namespace netket {

/**
  User-Define Hilbert space
*/

class CustomHilbert : public AbstractHilbert {
  const AbstractGraph &graph_;
  std::vector<double> local_;

  int nstates_;

  int size_;

 public:
  CustomHilbert(const AbstractGraph &graph,
                const std::vector<double> &localstates);

  bool IsDiscrete() const override;
  int LocalSize() const override;
  int Size() const override;
  std::vector<double> LocalStates() const override;
  const AbstractGraph &GetGraph() const noexcept override;

  void RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                  netket::default_random_engine &rgen) const override;

  void UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                  const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const override;
};

}  // namespace netket

#endif  // NETKET_CUSTOM_HILBERT_HPP
