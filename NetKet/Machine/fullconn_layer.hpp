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
#include "Lookup/lookup.hpp"
#include "Utils/all_utils.hpp"
#include "abstract_layer.hpp"

namespace netket {

template <typename Activation, typename T>
class FullyConnected : public AbstractLayer<T> {
  using VectorType = typename AbstractLayer<T>::VectorType;
  using MatrixType = typename AbstractLayer<T>::MatrixType;

  Activation activation_;  // activation function class

  bool usebias_;

  int in_size_;        // input size
  int out_size_;       // output size
  int npar_;           // number of parameters in layer
  MatrixType weight_;  // Weight parameters, W(in_size x out_size)
  VectorType bias_;    // Bias parameters, b(out_size x 1)
  VectorType z_;       // Linear term, z = W' * in + b
  VectorType din_;     // Derivative of the input of this layer.
                       // Note that input of this layer is also the output of
                       // previous layer

  std::size_t scalar_bytesize_;

 public:
  using StateType = typename AbstractLayer<T>::StateType;
  using LookupType = typename AbstractLayer<T>::LookupType;

  /// Constructor
  FullyConnected(const int input_size, const int output_size,
                 const bool use_bias = true)
      : activation_(),
        usebias_(use_bias),
        in_size_(input_size),
        out_size_(output_size) {
    Init();
  }

  explicit FullyConnected(const json &pars) : activation_() { Init(pars); }

  void Init(const json &pars) {
    in_size_ = FieldVal(pars, "Inputs");
    out_size_ = FieldVal(pars, "Outputs");

    usebias_ = FieldOrDefaultVal(pars, "UseBias", true);

    scalar_bytesize_ = sizeof(std::complex<double>);

    weight_.resize(in_size_, out_size_);
    bias_.resize(out_size_);
    din_.resize(in_size_);

    npar_ = in_size_ * out_size_;

    z_.resize(out_size_);

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

  void Init() {
    scalar_bytesize_ = sizeof(std::complex<double>);

    weight_.resize(in_size_, out_size_);
    bias_.resize(out_size_);
    din_.resize(in_size_);

    npar_ = in_size_ * out_size_;

    z_.resize(out_size_);

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

  void GetParameters(VectorType &pars, int start_idx) override {
    int k = start_idx;

    if (usebias_) {
      std::memcpy(pars.data() + k, bias_.data(), out_size_ * scalar_bytesize_);
      k += out_size_;
    }

    std::memcpy(pars.data() + k, weight_.data(),
                in_size_ * out_size_ * scalar_bytesize_);
  }

  void SetParameters(const VectorType &pars, int start_idx) override {
    int k = start_idx;

    if (usebias_) {
      std::memcpy(bias_.data(), pars.data() + k, out_size_ * scalar_bytesize_);

      k += out_size_;
    }

    std::memcpy(weight_.data(), pars.data() + k,
                in_size_ * out_size_ * scalar_bytesize_);
  }

  void InitLookup(const Eigen::VectorXd &v, LookupType &lt) override {
    if (lt.VectorSize() == 0) {
      lt.AddVector(out_size_);
    }
    lt.V(0) = (weight_.transpose() * v + bias_);
  }

  void UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    if (tochange.size() != 0) {
      for (std::size_t s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        lt.V(0) += weight_.row(sf) * (newconf[s] - v(sf));
      }
    }
  }

  void Forward(const VectorType &prev_layer_output,
               VectorType &output) override {
    // Linear term z = W' * in + b
    z_ = bias_;
    z_.noalias() += weight_.transpose() * prev_layer_output;

    // Apply activation function
    output.resize(out_size_);
    activation_(z_, output);
  }

  // Using lookup
  void Forward(const VectorType & /*prev_layer_output*/, const LookupType &lt,
               VectorType &output) override {
    z_ = lt.V(0);
    // Apply activation function
    output.resize(out_size_);
    activation_(z_, output);
  }

  void Backprop(const VectorType &prev_layer_output,
                const VectorType &this_layer_output,
                const VectorType &next_layer_data, VectorType &der,
                int start_idx) override {
    // After forward stage, m_z contains z = W' * in + b
    // Now we need to calculate d(L) / d(z) = [d(a) / d(z)] * [d(L) / d(a)]
    // d(L) / d(a) is computed in the next layer, contained in next_layer_data
    // The Jacobian matrix J = d(a) / d(z) is determined by the activation
    // function
    VectorType &dLz = z_;
    activation_.ApplyJacobian(z_, this_layer_output, next_layer_data, dLz);

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
    din_.noalias() = weight_ * dLz;
  }

  const VectorType &BackpropData() const override { return din_; }
};
}  // namespace netket

#endif
