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
//
// by S. Efthymiou, October 2018

#ifndef NETKET_ABSTRACT_MPS_HPP
#define NETKET_ABSTRACT_MPS_HPP

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Lookup/lookup.hpp"
#include "Utils/all_utils.hpp"

namespace netket {

template <typename T>
class AbstractMPS : public AbstractMachine<T> {
 public:
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using StateType = T;
  using LookupType = Lookup<T>;

  // Auxiliary function used for setting initial random parameters and adding
  // identities in every matrix
  virtual inline void SetParametersIdentity(const VectorType &pars) = 0;

  virtual void InitLookup(const std::vector<int> &v, LookupType &lt,
                          const int &start_ind) = 0;

  virtual void UpdateLookup(const std::vector<int> &v,
                            const std::vector<int> &tochange,
                            const std::vector<int> &newconf, LookupType &lt,
                            const int &start_ind) = 0;

  virtual T LogVal(const std::vector<int> &v) = 0;

  virtual inline T LogVal(const LookupType &lt, const int &start_ind) = 0;

  virtual T LogValDiff(const std::vector<int> &v,
                       const std::vector<int> &toflip,
                       const std::vector<int> &newconf, const LookupType &lt,
                       const int &start_ind) = 0;

  // No (k and lookup)-dependent version for SBS use
  virtual T LogValDiff(const std::vector<int> &v,
                       const std::vector<int> &toflip,
                       const std::vector<int> &newconf) = 0;

  // For the case of one spin flip (not required to know v)
  virtual T FastLogValDiff(const std::vector<int> &toflip,
                           const std::vector<int> &newconf,
                           const LookupType &lt, const int &start_ind) = 0;

  virtual VectorType DerLog(const std::vector<int> &v) = 0;

  virtual inline void to_jsonWeights(json &j) const = 0;

  virtual inline void from_jsonWeights(const json &pars) = 0;

  virtual ~AbstractMPS() {}
};
}  // namespace netket

#endif
