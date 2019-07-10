// Copyright 2018-2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#ifndef NETKET_SR_HPP
#define NETKET_SR_HPP

#include <Eigen/Core>

namespace netket {

// Generalized Stochastic Reconfiguration Updates
class SR {
 public:
  double diag_shift;
  bool use_iterative;
  bool use_cholesky;
  bool is_holomorphic;

  SR(double diagshift, bool use_iterative, bool use_cholesky,
     bool is_holomorphic);

  void ComputeUpdate(const Eigen::Ref<const Eigen::MatrixXcd> Oks,
                     const Eigen::Ref<const Eigen::VectorXcd> grad,
                     Eigen::Ref<Eigen::VectorXcd> deltaP);

 private:
  Eigen::MatrixXd Sreal_;
  Eigen::MatrixXcd Scomplex_;
};

}  // namespace netket

#endif
