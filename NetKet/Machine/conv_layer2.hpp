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

#ifndef NETKET_CONV2LAYER_HH
#define NETKET_CONV2LAYER_HH

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
class Convolutional2 : public AbstractLayer<T> {
  using VectorType = typename AbstractLayer<T>::VectorType;
  using MatrixType = typename AbstractLayer<T>::MatrixType;

  Activation activation_;  // activation function class

  bool usebias_;  // boolean to turn or off bias

  int nv_;            // number of visible units in the full network
  int in_channels_;   // number of input channels
  int in_size_;       // input size: should be multiple of no. of sites
  int out_channels_;  // number of output channels
  int out_size_;      // output size: should be multiple of no. of sites
  int npar_;          // number of parameters in layer
  int nparbare_;      // number of bare parameters in layer

  int dist_;         // Distance to include in one convolutional image
  int kernel_size_;  // Size of convolutional kernel (depends on dist_)
  std::vector<std::vector<int>>
      neighbours_;  // list of neighbours for each site
  std::vector<std::vector<int>>
      flipped_neighbours_;  // list of reverse neighbours for each site
  MatrixType kernels_;      // Weight parameters, W(in_size x out_size)
  MatrixType weight_;       // Weight parameters, W(in_size x out_size)
  VectorType bias_;         // Bias parameters, b(out_size x 1)

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
  Convolutional2(const Graph &graph, const int input_channel,
                 const int output_channel, const int dist = 1,
                 const bool use_bias = true)
      : activation_(),
        usebias_(use_bias),
        nv_(graph.Nsites()),
        in_channels_(input_channel),
        out_channels_(output_channel),
        dist_(dist) {
    Init(graph);
  }

  explicit Convolutional2(const Graph &graph, const json &pars)
      : activation_(), nv_(graph.Nsites()) {
    in_channels_ = FieldVal(pars, "InputChannels");
    in_size_ = in_channels_ * nv_;

    out_channels_ = FieldVal(pars, "OutputChannels");
    out_size_ = out_channels_ * nv_;

    dist_ = FieldVal(pars, "Distance");

    usebias_ = FieldOrDefaultVal(pars, "UseBias", true);

    Init(graph);
  }

  void Init(const Graph &graph) {
    // Construct neighbours_ kernel(k) will act on neighbours_[i][k]
    std::vector<std::vector<int>> adjlist;
    adjlist = graph.AdjacencyList();
    for (int i = 0; i < nv_; i++) {
      std::vector<int> neigh;
      neigh.push_back(i);
      for (int d = 0; d < dist_; d++) {
        size_t current = neigh.size();
        for (int n = 0; n < current; n++) {
          for (auto m : adjlist[neigh[n]]) {
            bool isin = false;
            for (size_t k = 0; k < neigh.size(); ++k) {
              if (neigh[k] == m) {
                isin = true;
              }
            }
            if (!isin) {
              neigh.push_back(m);
            }
          }
        }
      }
      neighbours_.push_back(neigh);
    }

    // Check that all sites have same number of neighbours
    kernel_size_ = neighbours_[0].size();
    for (int i = 1; i < nv_; i++) {
      if (neighbours_[i].size() != kernel_size_) {
        throw InvalidInputError(
            "number of neighbours of each site is not the same for chosen "
            "lattice");
      }
    }

    // Construct flipped_neighbours_
    // flipped_neighbours_[i][k] should be acted on by kernel(k)
    // Let neighbours_[i][k] = l
    // i.e. look for the direction k' where neighbours_[l][k'] = i
    // then neighbours_[i][k'] will be acted on by kernel(k)
    // so flipped_neighbours_[i][k] = neighbours_[i][k']
    for (int i = 0; i < nv_; ++i) {
      std::vector<int> flippedneigh;
      for (int k = 0; k < kernel_size_; ++k) {
        int l = neighbours_[i][k];
        for (int kp = 0; kp < kernel_size_; ++kp) {
          if (neighbours_[l][kp] == i) {
            flippedneigh.push_back(neighbours_[i][kp]);
          }
        }
      }
      flipped_neighbours_.push_back(flippedneigh);
    }

    z_.resize(out_channels_ * nv_);
    a_.resize(out_channels_ * nv_);
    kernels_.resize(in_channels_ * kernel_size_, out_channels_);
    weight_ = Eigen::MatrixXd::Zero(in_size_, out_size_);
    bias_.resize(out_size_);
    dw_.resize(in_size_, out_size_);
    db_.resize(out_size_);
    din_.resize(in_size_);

    npar_ = in_channels_ * kernel_size_ * out_channels_;

    if (usebias_) {
      npar_ += out_channels_;
    } else {
      bias_.setZero();
    }

    nparbare_ = in_size_ * out_size_;

    DerMatSymm_ = Eigen::MatrixXd::Zero(npar_, in_size_ * out_size_);
    int kpar = 0;

    for (int out = 0; out < out_channels_; ++out) {
      for (int in = 0; in < in_channels_; ++in) {
        for (int k = 0; k < kernel_size_; ++k) {
          for (int i = 0; i < nv_; ++i) {
            DerMatSymm_(kpar, (out * nv_ + i) * nv_ * in_channels_ + in * nv_ +
                                  neighbours_[i][k]) = 1;
          }
          ++kpar;
        }
      }
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if (mynode_ == 0) {
      std::cout << "Convolutional2 Layer: " << in_size_ << " --> " << out_size_
                << std::endl;
      std::cout << "# # InputChannels = " << in_channels_ << std::endl;
      std::cout << "# # OutputChannels = " << out_channels_ << std::endl;
      std::cout << "# # Filter Distance = " << dist_ << std::endl;
      std::cout << "# # Filter Size = " << kernel_size_ << std::endl;
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
        pars(k) = bias_(i);
        ++k;
      }
    }

    for (int j = 0; j < out_channels_; j++) {
      for (int i = 0; i < in_channels_ * kernel_size_; i++) {
        pars(k) = kernels_(i, j);
        ++k;
      }
    }
  }

  void SetParameters(const VectorType &pars, int start_idx) override {
    int k = start_idx;

    if (usebias_) {
      for (int i = 0; i < out_channels_; i++) {
        bias_(i) = pars(k);
        ++k;
      }
    }

    for (int j = 0; j < out_channels_; j++) {
      for (int i = 0; i < in_channels_ * kernel_size_; i++) {
        kernels_(i, j) = pars(k);
        ++k;
      }
    }

    SetBareParameters();
  }

  void SetBareParameters() {
    // Map bare biases to symmetrix bias

    // Map bare weights to symmetrix weights
    for (int out = 0; out < out_channels_; ++out) {
      for (int in = 0; in < in_channels_; ++in) {
        for (int i = 0; i < nv_; ++i) {
          for (int k = 0; k < kernel_size_; ++k) {
            weight_(neighbours_[i][k] + in * nv_, i + out * nv_) =
                kernels_(k + in * kernel_size_, out);
          }
        }
      }
    }
  }

  void InitLookup(const Eigen::VectorXd &v, LookupType &lt) override {
    if (lt.VectorSize() == 0) {
      lt.AddVector(out_size_);
    }
    lt.V(0) = (weight_.transpose() * v);
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
    z_.noalias() = weight_.transpose() * prev_layer_data;

    activation_(z_, a_);

    std::cout << std::setprecision(5);
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

  // void Backprop(const VectorType &prev_layer_data,
  //               const VectorType &next_layer_data) override {
  //   // Compute dL/dz
  //   VectorType &dLz = z_;
  //   activation_.ApplyJacobian(z_, a_, next_layer_data, dLz);
  //
  //   // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
  //   dw_.noalias() = prev_layer_data * dLz.transpose();
  //
  //   // Derivative for bias, d(L) / d(b) = d(L) / d(z)
  //   if (usebias_) {
  //     db_.noalias() = dLz;
  //   }
  //
  //   // Compute d(L) / d_in = W * [d(L) / d(z)]
  //   din_.noalias() = weight_ * dLz;
  // }
  //
  // void GetDerivative(VectorType &der, int start_idx) override {
  //   VectorType bareder(nparbare_);
  //
  //   int kk = 0;
  //   if (usebias_) {
  //     for (int i = 0; i < out_size_; ++i) {
  //       bareder(kk) = db_(i);
  //       ++kk;
  //     }
  //   }
  //
  //   for (int j = 0; j < out_size_; ++j) {
  //     for (int i = 0; i < in_size_; ++i) {
  //       bareder(kk) = dw_(i, j);
  //       ++kk;
  //     }
  //   }
  //
  //   VectorType symmder(npar_);
  //   symmder = DerMatSymm_ * bareder;
  //
  //   int k = start_idx;
  //
  //   for (int i = 0; i < npar_; ++i) {
  //     der(k) = symmder(i);
  //     ++k;
  //   }
  // }
};
}  // namespace netket

#endif
