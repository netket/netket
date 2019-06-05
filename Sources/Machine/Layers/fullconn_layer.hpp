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

#ifndef NETKET_FULLCONNLAYER_HH
#define NETKET_FULLCONNLAYER_HH

#include <Eigen/Dense>
#include <complex>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"
#include "abstract_layer.hpp"

namespace netket {

class FullyConnected : public AbstractLayer {
  bool usebias_;

  int in_size_;        // input size
  int out_size_;       // output size
  int npar_;           // number of parameters in layer
  MatrixType weight_;  // Weight parameters, W(in_size x out_size)
  VectorType bias_;    // Bias parameters, b(out_size x 1)
                       // Note that input of this layer is also the output of
                       // previous layer

  std::string name_;

  std::size_t scalar_bytesize_;

 public:
  /// Constructor
  FullyConnected(const int input_size, const int output_size,
                 const bool use_bias = false)
      : usebias_(use_bias), in_size_(input_size), out_size_(output_size) {
    Init();
  }

  void Init() {
    scalar_bytesize_ = sizeof(Complex);

    weight_.resize(in_size_, out_size_);
    bias_.resize(out_size_);

    npar_ = in_size_ * out_size_;

    if (usebias_) {
      npar_ += out_size_;
    } else {
      bias_.setZero();
    }

    name_ = "Fully Connected Layer";
  }

  std::string Name() const override { return name_; }

  void to_json(json &pars) const override {
    json layerpar;
    layerpar["Name"] = "FullyConnected";
    layerpar["UseBias"] = usebias_;
    layerpar["Inputs"] = in_size_;
    layerpar["Outputs"] = out_size_;
    layerpar["Bias"] = bias_;
    layerpar["Weight"] = weight_;

    pars["Layers"].push_back(layerpar);
  }

  void from_json(const json &pars) override {
    if (FieldExists(pars, "Weight")) {
      weight_ = pars["Weight"];
    } else {
      weight_.setZero();
    }
    if (FieldExists(pars, "Bias")) {
      bias_ = pars["Bias"];
    } else {
      bias_.setZero();
    }
  }

  void InitRandomPars(int seed, double sigma) override {
    VectorType par(npar_);

    netket::RandomGaussian(par, seed, sigma);

    SetParameters(par);
  }

  int Npar() const override { return npar_; }

  int Ninput() const override { return in_size_; }

  int Noutput() const override { return out_size_; }

  void GetParameters(VectorRefType pars) const override {
    int k = 0;
    if (usebias_) {
      std::memcpy(pars.data(), bias_.data(), out_size_ * scalar_bytesize_);
      k += out_size_;
    }

    std::memcpy(pars.data() + k, weight_.data(),
                in_size_ * out_size_ * scalar_bytesize_);
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    if (usebias_) {
      std::memcpy(bias_.data(), pars.data() + k, out_size_ * scalar_bytesize_);

      k += out_size_;
    }

    std::memcpy(weight_.data(), pars.data() + k,
                in_size_ * out_size_ * scalar_bytesize_);
  }

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

  // Feedforward
  void Forward(const VectorType &input, VectorType &output) override {
    output = bias_;
    output.noalias() += weight_.transpose() * input;
  }

  // Updates theta given the input v, the change in the input (input_changes and
  // prev_input)
  inline void UpdateOutput(const VectorType &v,
                           const std::vector<int> &input_changes,
                           const VectorType &new_input,
                           VectorType &new_output) {
    const int num_of_changes = input_changes.size();
    for (int s = 0; s < num_of_changes; s++) {
      const int sf = input_changes[s];
      new_output += weight_.row(sf) * (new_input(s) - v(sf));
    }
  }

  // Computes derivative.
  void Backprop(const VectorType &prev_layer_output,
                const VectorType & /*this_layer_output*/,
                const VectorType &dout, VectorType &din,
                VectorRefType der) override {
    // dout = d(L) / d(z)
    // Derivative for bias, d(L) / d(b) = d(L) / d(z)
    int k = 0;

    if (usebias_) {
      Eigen::Map<VectorType> der_b{der.data() + k, out_size_};

      der_b.noalias() = dout;
      k += out_size_;
    }

    // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
    Eigen::Map<MatrixType> der_w{der.data() + k, in_size_, out_size_};

    der_w.noalias() = prev_layer_output * dout.transpose();

    // Compute d(L) / d_in = W * [d(L) / d(z)]
    din.noalias() = weight_ * dout;
  }
};
}  // namespace netket

#endif
