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
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"
#include "abstract_layer.hpp"

namespace netket {

class SumOutput : public AbstractLayer {
  int in_size_;   // input size: should be multiple of no. of sites
  int out_size_;  // output size: should be multiple of no. of sites

  VectorType z_;  // Linear term, z = W' * in + b

  std::string name_;

 public:
  /// Constructor
  explicit SumOutput(int in_size) : in_size_(in_size) {
    out_size_ = 1;

    Init();
  }

  // TODO remove
  /// Constructor
  explicit SumOutput(const json &pars) {
    in_size_ = FieldVal(pars, "Inputs");

    out_size_ = 1;

    Init();
  }

  void Init() {
    z_.resize(out_size_);

    name_ = "Sum Output Layer";
  }

  std::string Name() const override { return name_; }

  void InitRandomPars(int /*seed*/, double /*sigma*/) override {}

  int Npar() const override { return 0; }

  int Ninput() const override { return in_size_; }

  int Noutput() const override { return out_size_; }

  void GetParameters(VectorRefType /*pars*/) const override {}

  void SetParameters(VectorConstRefType /*pars*/) override {}

  void UpdateLookup(const VectorType &input,
                    const std::vector<int> &input_changes,
                    const VectorType &new_input, const VectorType &output,
                    std::vector<int> &output_changes,
                    VectorType &new_output) override {
    const int num_of_changes = input_changes.size();
    if (num_of_changes == in_size_) {
      output_changes.resize(out_size_);
      new_output.resize(out_size_);
      Forward(new_input, new_output);
    } else if (num_of_changes > 0) {
      output_changes.resize(out_size_);
      new_output = output;
      UpdateOutput(input, input_changes, new_input, new_output);
    } else {
      output_changes.resize(0);
      new_output.resize(0);
    }
  }

  void Forward(const VectorType &input, VectorType &output) override {
    output(0) = input.sum();
  }

  inline void UpdateOutput(const VectorType &v,
                           const std::vector<int> &input_changes,
                           const VectorType &new_input,
                           VectorType &new_output) {
    const int num_of_changes = input_changes.size();
    for (int s = 0; s < num_of_changes; s++) {
      const int sf = input_changes[s];
      new_output(0) += (new_input(s) - v(sf));
    }
  }

  void Backprop(const VectorType & /*prev_layer_output*/,
                const VectorType & /*this_layer_output*/,
                const VectorType &dout, VectorType &din,
                VectorRefType /*der*/) override {
    din.resize(in_size_);
    din.setConstant(dout(0));
  }

  void to_json(json &pars) const override {
    json layerpar;
    layerpar["Name"] = "Sum";
    layerpar["Inputs"] = in_size_;
    layerpar["Outputs"] = out_size_;

    pars["Layers"].push_back(layerpar);
  }

  void from_json(const json & /*j*/) override {}
};
}  // namespace netket

#endif
