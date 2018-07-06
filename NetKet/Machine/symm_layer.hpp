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

#ifndef NETKET_SYMMLAYER_HH
#define NETKET_SYMMLAYER_HH

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

template <typename Activation, typename T>
class Symmetric : public AbstractLayer<T> {
  using VectorType = typename AbstractLayer<T>::VectorType;
  using MatrixType = typename AbstractLayer<T>::MatrixType;

  Activation activation_;  // activation function class

  bool usebias_;  // boolean to turn or off bias

  std::vector<std::vector<int>> permtable_;
  int permsize_;

  int nv_;            // number of visible units in the full network
  int in_channels_;   // number of input channels
  int in_size_;       // input size: should be multiple of no. of sites
  int out_channels_;  // number of output channels
  int out_size_;      // output size: should be multiple of no. of sites
  int npar_;          // number of parameters in layer
  int nparbare_;      // number of bare parameters in layer

  MatrixType wsymm_;   // Weight parameters, W(in_size x out_size)
  VectorType bsymm_;   // Weight parameters, W(in_size x out_size)
  MatrixType weight_;  // Weight parameters, W(in_size x out_size)
  VectorType bias_;    // Bias parameters, b(out_size x 1)

  MatrixType dw_;  // Derivative of weights
  VectorType db_;  // Derivative of bias

  VectorType z_;    // Linear term, z = W' * in + b
  VectorType a_;    // Output of this layer, a = act(z)
  VectorType din_;  // Derivative of the input of this layer.
  // Note that input of this layer is also the output of
  // previous layer

  MatrixType DerMatSymm_;

  int mynode_;

 public:
  using StateType = typename AbstractLayer<T>::StateType;
  using LookupType = typename AbstractLayer<T>::LookupType;

  /// Constructor
  Symmetric(const Graph &graph, const int input_channel,
            const int output_channel, const bool use_bias = true)
      : activation_(),
        usebias_(use_bias),
        nv_(graph.Nsites()),
        in_channels_(input_channel),
        out_channels_(output_channel) {
    Init(graph);
  }

  explicit Symmetric(const Graph &graph, const json &pars)
      : activation_(), nv_(graph.Nsites()) {
    in_channels_ = FieldVal(pars, "InputChannels");
    in_size_ = in_channels_ * nv_;

    out_channels_ = FieldVal(pars, "OutputChannels");
    out_size_ = out_channels_ * nv_;

    usebias_ = FieldOrDefaultVal(pars, "UseBias", true);

    Init(graph);
  }

  void Init(const Graph &graph) {
    permtable_ = graph.SymmetryTable();
    permsize_ = permtable_.size();

    z_.resize(out_channels_ * nv_);
    a_.resize(out_channels_ * nv_);
    wsymm_.resize(in_channels_ * nv_, out_channels_);
    bsymm_.resize(out_channels_);
    weight_ = Eigen::MatrixXd::Zero(in_size_, out_size_);
    bias_.resize(out_size_);
    dw_.resize(in_size_, out_size_);
    db_.resize(out_size_);
    din_.resize(in_size_);

    npar_ = in_channels_ * nv_ * out_channels_;
    nparbare_ = in_size_ * out_size_;

    if (usebias_) {
      npar_ += out_channels_;
      nparbare_ += out_size_;
    } else {
      bias_.setZero();
    }

    DerMatSymm_ = Eigen::MatrixXd::Zero(npar_, nparbare_);
    int kpar = 0;

    if (usebias_) {
      for (int out = 0; out < out_channels_; ++out) {
        for (int k = 0; k < nv_; ++k) {
          DerMatSymm_(kpar, k + out * nv_) = 1;
        }
        ++kpar;
      }
    }

    for (int out = 0; out < out_channels_; ++out) {
      for (int in = 0; in < in_channels_; ++in) {
        for (int k = 0; k < nv_; ++k) {
          for (int i = 0; i < nv_; ++i) {
            DerMatSymm_(kpar, out_size_ + (out * nv_ + i) * nv_ * in_channels_ +
                                  in * nv_ + permtable_[i][k]) = 1;
          }
          ++kpar;
        }
      }
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (mynode_ == 0) {
      std::cout << "Symmetric Layer: " << in_size_ << " --> " << out_size_
                << std::endl;
      std::cout << "# # InputChannels = " << in_channels_ << std::endl;
      std::cout << "# # OutputChannels = " << out_channels_ << std::endl;
      std::cout << "# # UseBias = " << usebias_ << std::endl;
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
      for (int i = 0; i < out_channels_; i++) {
        pars(k) = bsymm_(i);
        ++k;
      }
    }

    for (int j = 0; j < out_channels_; j++) {
      for (int i = 0; i < in_size_; i++) {
        pars(k) = wsymm_(i, j);
        ++k;
      }
    }
  }

  void SetParameters(const VectorType &pars, int start_idx) override {
    int k = start_idx;

    if (usebias_) {
      for (int i = 0; i < out_channels_; i++) {
        bsymm_(i) = pars(k);
        ++k;
      }
    }

    for (int j = 0; j < out_channels_; j++) {
      for (int i = 0; i < in_size_; i++) {
        wsymm_(i, j) = pars(k);
        ++k;
      }
    }

    SetBareParameters();
  }

  void SetBareParameters() {
    // Map bare biases to symmetrix bias
    if (usebias_) {
      int k = 0;
      for (int i = 0; i < out_channels_; i++) {
        for (int j = 0; j < nv_; ++j) {
          bias_(k) = bsymm_(i);
          ++k;
        }
      }
    }

    // Map bare weights to symmetrix weights
    for (int out = 0; out < out_channels_; ++out) {
      for (int in = 0; in < in_channels_; ++in) {
        for (int k = 0; k < nv_; ++k) {
          for (int i = 0; i < nv_; ++i) {
            weight_(permtable_[i][k] + in * nv_, i + out * nv_) =
                wsymm_(k + in * nv_, out);
          }
        }
      }
    }
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

  void Forward(const VectorType &prev_layer_data) override {
    z_ = bias_;
    z_.noalias() += weight_.transpose() * prev_layer_data;

    activation_(z_, a_);
  }

  // Using lookup
  void Forward(const VectorType & /*prev_layer_data*/,
               const LookupType &lt) override {
    z_ = lt.V(0);
    // Apply activation function
    activation_(z_, a_);
  }

  VectorType Output() const override { return a_; }

  void Backprop(const VectorType &prev_layer_data,
                const VectorType &next_layer_data, VectorType &der,
                int start_idx) override {
    // Compute dL/dz
    VectorType &dLz = z_;
    activation_.ApplyJacobian(z_, a_, next_layer_data, dLz);

    // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
    VectorType bareder(nparbare_);
    int k = 0;
    if (usebias_) {
      Eigen::Map<VectorType> der_b{bareder.data() + k, out_size_};

      der_b.noalias() = dLz;

      k += out_size_;
    }

    // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
    Eigen::Map<MatrixType> der_w{bareder.data() + k, in_size_, out_size_};

    der_w.noalias() = prev_layer_data * dLz.transpose();

    Eigen::Map<VectorType> der_all{der.data() + start_idx, npar_};
    der_all.noalias() = DerMatSymm_ * bareder;

    // Compute d(L) / d_in = W * [d(L) / d(z)]
    din_.noalias() = weight_ * dLz;
  }

  const VectorType &BackpropData() const override { return din_; }
};
}  // namespace netket

#endif
