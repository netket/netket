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

#ifndef NETKET_SUMOUTPUT_HH
#define NETKET_SUMOUTPUT_HH

#include <time.h>
#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <fstream>
#include <random>
#include <vector>
#include "Lookup/lookup.hpp"
#include "Utils/all_utils.hpp"
#include "abstract_layer.hpp"

namespace netket {

template <typename T>
class SumOutput : public AbstractLayer<T> {
  using VectorType = typename AbstractLayer<T>::VectorType;
  using MatrixType = typename AbstractLayer<T>::MatrixType;

  int in_size_;   // input size: should be multiple of no. of sites
  int out_size_;  // output size: should be multiple of no. of sites

  VectorType z_;  // Linear term, z = W' * in + b

  VectorType din_;  // Derivative of the input of this layer.
  // Note that input of this layer is also the output of
  // previous layer

  int mynode_;

 public:
  using StateType = typename AbstractLayer<T>::StateType;
  using LookupType = typename AbstractLayer<T>::LookupType;

  /// Constructor
  explicit SumOutput(const json &pars) {
    in_size_ = FieldVal(pars, "Inputs");

    out_size_ = 1;

    Init();
  }

  void Init() {
    din_.resize(in_size_);
    din_.setConstant(1);
    z_.resize(out_size_);

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (mynode_ == 0) {
      std::cout << "Sum Output Layer: " << in_size_ << " --> " << out_size_
                << std::endl;
    }
  }

  void InitRandomPars(int /*seed*/, double /*sigma*/) override {}

  int Npar() const override { return 0; }

  int Ninput() const override { return in_size_; }

  int Noutput() const override { return out_size_; }

  void GetParameters(VectorType & /*pars*/, int /*start_idx*/) override {}

  void SetParameters(const VectorType & /*pars*/, int /*start_idx*/) override {}

  void InitLookup(const Eigen::VectorXd & /*v*/, LookupType & /*lt*/) override {
  }

  void UpdateLookup(const Eigen::VectorXd & /*v*/,
                    const std::vector<int> & /*tochange*/,
                    const std::vector<double> & /*newconf*/,
                    LookupType & /*lt*/) override {}

  void Forward(const VectorType &prev_layer_data) override {
    z_(0) = prev_layer_data.sum();
  }

  // Using lookup
  void Forward(const VectorType &prev_layer_data,
               const LookupType & /*lt*/) override {
    z_(0) = prev_layer_data.sum();
  }

  VectorType Output() const override { return z_; }

  void Backprop(const VectorType & /*prev_layer_data*/,
                const VectorType & /*next_layer_data*/, VectorType & /*der*/,
                int /*start_idx*/) override {}

  const VectorType &BackpropData() const override { return din_; }
};
}  // namespace netket

#endif
