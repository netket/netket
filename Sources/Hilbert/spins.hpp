// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
// Copyright 2018 Tom Westerhout
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

#include "Graph/abstract_graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"
#include "Utils/random_utils.hpp"

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
 public:
  Spin(const AbstractGraph &graph, double S);
  Spin(const AbstractGraph &graph, double S, double totalSz);

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

 private:
  void Init();
  inline void SetConstraint(double totalS);

  const AbstractGraph &graph_;
  double S_;
  double totalS_;
  bool constraintSz_;
  std::vector<double> local_;
  int nstates_;
  int nspins_;
};

}  // namespace netket

#endif  // NETKET_SPIN_HPP
