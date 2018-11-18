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
#include <vector>
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"
#include "abstract_layer.hpp"

namespace netket {

template <typename T>
class FullyConnected : public AbstractLayer<T> {
  using VectorType = typename AbstractLayer<T>::VectorType;
  using MatrixType = typename AbstractLayer<T>::MatrixType;
  using VectorRefType = typename AbstractLayer<T>::VectorRefType;
  using VectorConstRefType = typename AbstractLayer<T>::VectorConstRefType;

  std::shared_ptr<const AbstractActivation>
      activation_;  // activation function class

  bool usebias_;

  int in_size_;        // input size
  int out_size_;       // output size
  int npar_;           // number of parameters in layer
  MatrixType weight_;  // Weight parameters, W(in_size x out_size)
  VectorType bias_;    // Bias parameters, b(out_size x 1)
                       // Note that input of this layer is also the output of
                       // previous layer

  std::size_t scalar_bytesize_;

 public:
  using StateType = typename AbstractLayer<T>::StateType;
  using LookupType = typename AbstractLayer<T>::LookupType;

  /// Constructor
  FullyConnected(std::shared_ptr<const AbstractActivation> activation,
                 const int input_size, const int output_size,
                 const bool use_bias = false)
      : activation_(activation),
        usebias_(use_bias),
        in_size_(input_size),
        out_size_(output_size) {
    Init();
  }

  void Init() {
    scalar_bytesize_ = sizeof(std::complex<double>);

    weight_.resize(in_size_, out_size_);
    bias_.resize(out_size_);

    npar_ = in_size_ * out_size_;

    if (usebias_) {
      npar_ += out_size_;
    } else {
      bias_.setZero();
    }

    std::string buffer = "";

    InfoMessage(buffer) << "Fully Connected Layer " << in_size_ << " --> "
                        << out_size_ << std::endl;
    InfoMessage(buffer) << "# # UseBias = " << usebias_ << std::endl;
  }

  void to_json(json &pars) const override {
    json layerpar;
    layerpar["Name"] = "FullyConnected";
    layerpar["UseBias"] = usebias_;
    layerpar["Inputs"] = in_size_;
    layerpar["Outputs"] = out_size_;
    layerpar["Bias"] = bias_;
    layerpar["Weight"] = weight_;

    pars["Machine"]["Layers"].push_back(layerpar);
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

    SetParameters(par, 0);
  }

  int Npar() const override { return npar_; }

  int Ninput() const override { return in_size_; }

  int Noutput() const override { return out_size_; }

  void GetParameters(VectorRefType pars, int start_idx) const override {
    int k = start_idx;

    if (usebias_) {
      std::memcpy(pars.data() + k, bias_.data(), out_size_ * scalar_bytesize_);
      k += out_size_;
    }

    std::memcpy(pars.data() + k, weight_.data(),
                in_size_ * out_size_ * scalar_bytesize_);
  }

  void SetParameters(VectorConstRefType pars, int start_idx) override {
    int k = start_idx;

    if (usebias_) {
      std::memcpy(bias_.data(), pars.data() + k, out_size_ * scalar_bytesize_);

      k += out_size_;
    }

    std::memcpy(weight_.data(), pars.data() + k,
                in_size_ * out_size_ * scalar_bytesize_);
  }

  void InitLookup(const VectorType &v, LookupType &lt,
                  VectorType &output) override {
    lt.resize(1);
    lt[0].resize(out_size_);

    Forward(v, lt, output);
  }

  void UpdateLookup(const VectorType &input,
                    const std::vector<int> &input_changes,
                    const VectorType &new_input, LookupType &theta,
                    const VectorType & /*output*/,
                    std::vector<int> &output_changes,
                    VectorType &new_output) override {
    const int num_of_changes = input_changes.size();
    if (num_of_changes == in_size_) {
      output_changes.resize(out_size_);
      new_output.resize(out_size_);
      Forward(new_input, theta, new_output);
    } else if (num_of_changes > 0) {
      UpdateTheta(input, input_changes, new_input, theta);
      output_changes.resize(out_size_);
      new_output.resize(out_size_);
      Forward(theta, new_output);
    } else {
      output_changes.resize(0);
      new_output.resize(0);
    }
  }

  void UpdateLookup(const Eigen::VectorXd &input,
                    const std::vector<int> &tochange,
                    const std::vector<double> &newconf, LookupType &theta,
                    const VectorType & /*output*/,
                    std::vector<int> &output_changes,
                    VectorType &new_output) override {
    const int num_of_changes = tochange.size();
    if (num_of_changes > 0) {
      UpdateTheta(input, tochange, newconf, theta);
      output_changes.resize(out_size_);
      new_output.resize(out_size_);
      Forward(theta, new_output);
    } else {
      output_changes.resize(0);
      new_output.resize(0);
    }
  }

  // Feedforward
  void Forward(const VectorType &prev_layer_output, LookupType &theta,
               VectorType &output) override {
    LinearTransformation(prev_layer_output, theta);
    NonLinearTransformation(theta, output);
  }

  // Feedforward Using lookup
  void Forward(const LookupType &theta, VectorType &output) override {
    // Apply activation function
    NonLinearTransformation(theta, output);
  }

  // Applies the linear transformation
  inline void LinearTransformation(const VectorType &input, LookupType &theta) {
    theta[0] = bias_;
    theta[0].noalias() += weight_.transpose() * input;
  }

  // Applies the nonlinear transformation
  inline void NonLinearTransformation(const LookupType &theta,
                                      VectorType &output) {
    activation_->operator()(theta[0], output);
  }

  // Updates theta given the input v, the change in the input (input_changes and
  // prev_input)
  inline void UpdateTheta(const VectorType &v,
                          const std::vector<int> &input_changes,
                          const VectorType &new_input, LookupType &theta) {
    const int num_of_changes = input_changes.size();
    for (int s = 0; s < num_of_changes; s++) {
      const int sf = input_changes[s];
      theta[0] += weight_.row(sf) * (new_input(s) - v(sf));
    }
  }

  // Updates theta given the previous input prev_input and the change in the
  // input (tochange and  newconf)
  inline void UpdateTheta(const VectorType &prev_input,
                          const std::vector<int> &tochange,
                          const std::vector<double> &newconf,
                          LookupType &theta) {
    const int num_of_changes = tochange.size();
    for (int s = 0; s < num_of_changes; s++) {
      const int sf = tochange[s];
      theta[0] += weight_.row(sf) * (newconf[s] - prev_input(sf));
    }
  }

  // Computes derivative.
  void Backprop(const VectorType &prev_layer_output,
                const VectorType &this_layer_output,
                const LookupType &this_layer_theta, const VectorType &dout,
                VectorType &din, VectorType &der, int start_idx) override {
    // After forward stage, m_z contains z = W' * in + b
    // Now we need to calculate d(L) / d(z) = [d(a) / d(z)] * [d(L) / d(a)]
    // d(L) / d(a) is computed in the next layer, contained in next_layer_data
    // The Jacobian matrix J = d(a) / d(z) is determined by the activation
    // function
    VectorType dLz(out_size_);
    activation_->ApplyJacobian(this_layer_theta[0], this_layer_output, dout,
                               dLz);

    // Now dLz contains d(L) / d(z)
    // Derivative for bias, d(L) / d(b) = d(L) / d(z)
    int k = start_idx;

    if (usebias_) {
      Eigen::Map<VectorType> der_b{der.data() + k, out_size_};

      der_b.noalias() = dLz;
      k += out_size_;
    }

    // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
    Eigen::Map<MatrixType> der_w{der.data() + k, in_size_, out_size_};

    der_w.noalias() = prev_layer_output * dLz.transpose();

    // Compute d(L) / d_in = W * [d(L) / d(z)]
    din.noalias() = weight_ * dLz;
  }
};
}  // namespace netket

#endif
