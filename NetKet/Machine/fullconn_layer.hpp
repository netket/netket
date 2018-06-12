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
#include <Lookup/lookup.hpp>
#include <complex>
#include <fstream>
#include <netket.hpp>
#include <random>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace netket {

using namespace std;
using namespace Eigen;

template <typename Activation, typename T>
class FullyConnected : public AbstractLayer<T> {
  using VectorType = typename AbstractLayer<T>::VectorType;
  using MatrixType = typename AbstractLayer<T>::MatrixType;

  int in_size_;       // input size
  int out_size_;      // output size
  int npar_;          // number of parameters in layer
  MatrixType weight_; // Weight parameters, W(in_size x out_size)
  VectorType bias_;   // Bias parameters, b(out_size x 1)
  MatrixType dw_;     // Derivative of weights
  VectorType db_;     // Derivative of bias
  VectorType z_;      // Linear term, z = W' * in + b
  VectorType a_;      // Output of this layer, a = act(z)
  VectorType din_;    // Derivative of the input of this layer.
                      // Note that input of this layer is also the output of
                      // previous layer

  int mynode_;
public:
  using StateType = typename AbstractLayer<T>::StateType;
  using LookupType = typename AbstractLayer<T>::LookupType;

  /// Constructor
  FullyConnected(int input_size, int output_size)
      : in_size_(input_size), out_size_(output_size) {
    Init();
  }
  void Init() {
    weight_.resize(in_size_, out_size_);
    bias_.resize(out_size_);
    dw_.resize(in_size_, out_size_);
    db_.resize(out_size_);
    npar_ = in_size_ * out_size_ + out_size_;

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (mynode_ == 0) {
      std::cout << ": Fully Connected Layer " << in_size_ << " --> " << out_size_ << std::endl;
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

    for (int i = 0; i < out_size_; ++i) {
      pars(k) = bias_(i);
      ++k;
    }

    for (int i = 0; i < in_size_; ++i) {
      for (int j = 0; j < out_size_; ++j) {
        pars(k) = weight_(i, j);
        ++k;
      }
    }
  }

  void SetParameters(const VectorType &pars, int start_idx) override {
    int k = start_idx;

    for (int i = 0; i < out_size_; ++i) {
      bias_(i) = pars(k);
      ++k;
    }

    for (int i = 0; i < in_size_; ++i) {
      for (int j = 0; j < out_size_; ++j) {
        weight_(i, j) = pars(k);
        ++k;
      }
    }
  }

  void InitLookup(const VectorXd &v, LookupType &lt) override {
    if (lt.VectorSize() == 0) {
      lt.AddVector(out_size_);
    }
    lt.V(0) = (weight_.transpose() * v + bias_);
  }

  void UpdateLookup(const VectorXd &v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    if (tochange.size() != 0) {
      for (int s = 0; s < tochange.size(); s++) {
        const int sf = tochange[s];
        lt.V(0) += weight_.row(sf) * (newconf[s] - v(sf));
      }
    }
  }

  void Forward(const VectorType &prev_layer_data) override {
    // Linear term z = W' * in + b
    z_.resize(out_size_);
    z_.noalias() = weight_.transpose() * prev_layer_data;
    z_.colwise() += bias_;

    // Apply activation function
    a_.resize(out_size_);
    Activation::activate(z_, a_);
  }

  // Using lookup
  void Forward(const VectorType &prev_layer_data, const LookupType &lt) override {
    z_.resize(out_size_);
    z_ = lt.V(0);
    // Apply activation function
    a_.resize(out_size_);
    Activation::activate(z_, a_);
  }

  VectorType Output() const override { return a_; }

  void Backprop(const VectorType &prev_layer_data,
                const VectorType &next_layer_data) override {

    // After forward stage, m_z contains z = W' * in + b
    // Now we need to calculate d(L) / d(z) = [d(a) / d(z)] * [d(L) / d(a)]
    // d(L) / d(a) is computed in the next layer, contained in next_layer_data
    // The Jacobian matrix J = d(a) / d(z) is determined by the activation
    // function
    VectorType &dLz = z_;
    Activation::apply_jacobian(z_, a_, next_layer_data, dLz);

    // Now dLz contains d(L) / d(z)
    // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
    dw_.noalias() = prev_layer_data * dLz.transpose();

    // Derivative for bias, d(L) / d(b) = d(L) / d(z)
    db_.noalias() = dLz;

    // Compute d(L) / d_in = W * [d(L) / d(z)]
    din_.resize(in_size_);
    din_.noalias() = weight_ * dLz;
  }

  const VectorType &Backprop_data() const override { return din_; }

  void GetDerivative(VectorType &der, int start_idx) override {
    int k = start_idx;
    for (int i = 0; i < out_size_; ++i) {
      der(k) = db_(i);
      ++k;
    }
    for (int i = 0; i < in_size_; ++i) {
      for (int j = 0; j < out_size_; ++j) {
        der(k) = dw_(i, j);
        ++k;
      }
    }
  }
};
} // namespace netket

#endif
