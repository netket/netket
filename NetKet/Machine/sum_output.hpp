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

  // Note that input of this layer is also the output of
  // previous layer

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
    z_.resize(out_size_);

    std::string buffer = "";
    InfoMessage(buffer) << "Sum Output Layer: " << in_size_ << " --> "
                        << out_size_ << std::endl;
  }

  void InitRandomPars(int /*seed*/, double /*sigma*/) override {}

  int Npar() const override { return 0; }

  int Ninput() const override { return in_size_; }

  int Noutput() const override { return out_size_; }

  void GetParameters(VectorType & /*pars*/, int /*start_idx*/) const override {}

  void SetParameters(const VectorType & /*pars*/, int /*start_idx*/) override {}

  void ForwardUpdate(const VectorType &input,
                     const std::vector<int> &input_changes,
                     const VectorType &prev_input, VectorType &theta,
                     VectorType &output, std::vector<int> &output_changes,
                     VectorType &prev_output) override {
    if (int(input_changes.size()) == in_size_) {
      LinearTransformation(input, theta);
    } else {
      UpdateTheta(input, input_changes, prev_input, theta);
    }
    UpdateOutput(theta, input_changes, output, output_changes, prev_output);
  }

  void ForwardUpdate(const Eigen::VectorXd &prev_input,
                     const std::vector<int> &tochange,
                     const std::vector<double> &newconf, VectorType &theta,
                     VectorType &output, std::vector<int> &output_changes,
                     VectorType &prev_output) override {
    UpdateTheta(prev_input, tochange, newconf, theta);
    UpdateOutput(theta, tochange, output, output_changes, prev_output);
  }

  void Forward(const VectorType &prev_layer_output, VectorType &theta,
               VectorType &output) override {
    LinearTransformation(prev_layer_output, theta);
    NonLinearTransformation(theta, output);
  }

  // Using lookup
  void Forward(const VectorType &theta, VectorType &output) override {
    // Apply activation function
    NonLinearTransformation(theta, output);
  }

  inline void LinearTransformation(const VectorType &input, VectorType &theta) {
    theta(0) = input.sum();
  }

  inline void NonLinearTransformation(const VectorType &theta,
                                      VectorType &output) {
    output(0) = theta(0);
  }

  inline void UpdateOutput(const VectorType &theta,
                           const std::vector<int> & /*input_changes*/,
                           VectorType &output,
                           std::vector<int> & /*output_changes*/,
                           VectorType & /*prev_output*/) {
    NonLinearTransformation(theta, output);
  }

  inline void UpdateTheta(const VectorType &v,
                          const std::vector<int> &input_changes,
                          const VectorType &prev_input, VectorType &theta) {
    const int num_of_changes = input_changes.size();
    for (int s = 0; s < num_of_changes; s++) {
      const int sf = input_changes[s];
      theta(0) += (v(sf) - prev_input(s));
    }
  }

  inline void UpdateTheta(const VectorType &prev_input,
                          const std::vector<int> &tochange,
                          const std::vector<double> &newconf,
                          VectorType &theta) {
    const int num_of_changes = tochange.size();
    for (int s = 0; s < num_of_changes; s++) {
      const int sf = tochange[s];
      theta(0) += (newconf[s] - prev_input(sf));
    }
  }

  void Backprop(const VectorType & /*prev_layer_output*/,
                const VectorType & /*this_layer_output*/,
                const VectorType & /*this_layer_theta*/,
                const VectorType &next_layer_data, VectorType &din,
                VectorType & /*der*/, int /*start_idx*/) override {
    din.resize(in_size_);
    din.setConstant(next_layer_data(0));
  }

  void to_json(json &pars) const override {
    json layerpar;
    layerpar["Name"] = "Sum";
    layerpar["Inputs"] = in_size_;
    layerpar["Outputs"] = out_size_;

    pars["Machine"]["Layers"].push_back(layerpar);
  }

  void from_json(const json & /*j*/) override {}
};
}  // namespace netket

#endif
