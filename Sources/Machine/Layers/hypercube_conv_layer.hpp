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

#ifndef NETKET_HYPERCUBECONVLAYER_HH
#define NETKET_HYPERCUBECONVLAYER_HH

#include <time.h>
#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include "Graph/abstract_graph.hpp"
#include "Utils/all_utils.hpp"
#include "Utils/lookup.hpp"
#include "abstract_layer.hpp"

namespace netket {
/** Convolutional layer with spin 1/2 hidden units.
 Important: In order for this to work correctly, VectorType and MatrixType must
 be column major.
 */
class ConvolutionalHypercube : public AbstractLayer {
  static_assert(!MatrixType::IsRowMajor, "MatrixType must be column-major");

  bool usebias_;  // boolean to turn or off bias

  int nv_;            // number of units in the input image
  int nout_;          // number of units in the output image
  int lin_;           // length of input hypercube
  int lout_;          // length of output hypercube
  int dim_;           // dimension of hypercube
  int in_channels_;   // number of input channels
  int in_size_;       // input size = in_channels_ * nv_;
  int out_channels_;  // number of output channels
  int out_size_;      // output size = out_channels_ * nout_;
  int npar_;          // number of parameters in layer

  int stride_;         // convolution stride
  int kernel_length_;  // length of kernel
  int kernel_size_;    // Size of convolutional kernel
  std::vector<std::vector<int>>
      neighbours_;  // list of neighbours for each site
  std::vector<std::vector<int>>
      flipped_nodes_;   // list of reverse neighbours for each site
  MatrixType kernels_;  // Weight parameters, W(in_size x out_size)
  VectorType bias_;     // Bias parameters, b(out_size x 1)

  std::string name_;

  MatrixType lowered_image_;
  MatrixType lowered_image2_;
  MatrixType lowered_der_;
  MatrixType flipped_kernels_;

 public:
  /// Constructor
  ConvolutionalHypercube(const int length, const int dim,
                         const int input_channels, const int output_channels,
                         const int stride = 1, const int kernel_length = 2,
                         const bool use_bias = true)
      : usebias_(use_bias),
        lin_(length),
        dim_(dim),
        in_channels_(input_channels),
        out_channels_(output_channels),
        stride_(stride),
        kernel_length_(kernel_length) {
    // Compatibility checks
    // Check that stride_ is compatible with input image length lin_
    if (lin_ % stride_ != 0) {
      throw InvalidInputError(
          "Stride size is incompatiple with input image size: they should be "
          "commensurate");
    }
    // Check that kernel_length_ is smaller than or equal input image length_
    if (kernel_length_ > lin_) {
      throw InvalidInputError(
          "kernel_length must be at most as large as input image length, "
          "length");
    }

    // Compute the image sizes
    lout_ = lin_ / stride_;
    nv_ = 1;
    nout_ = 1;
    kernel_size_ = 1;
    for (int i = 0; i < dim_; ++i) {
      nv_ *= lin_;
      nout_ *= lout_;
      kernel_size_ *= kernel_length_;
    }
    in_size_ = in_channels_ * nv_;
    out_size_ = out_channels_ * nout_;

    Init();
  }

  void Init() {
    // Construct neighbourhood of all nodes to be acted on by a kernel, i.e.
    // kernel(k) will act on the node neighbours_[i][k] of the
    // input image to give a value at node i in the output image.
    std::vector<Eigen::VectorXi> trans;
    for (int i = 0; i < kernel_size_; ++i) {
      trans.push_back(Site2Coord(i, kernel_length_));
    }
    for (int i = 0; i < nout_; ++i) {
      std::vector<int> neigh;
      Eigen::VectorXi coord = Site2Coord(i, lout_) * stride_;
      for (auto t : trans) {
        Eigen::VectorXi newcoord(dim_);
        for (int d = 0; d < dim_; ++d) {
          newcoord(d) = (coord(d) + t(d)) % lin_;
        }
        neigh.push_back(Coord2Site(newcoord, lin_));
      }
      neighbours_.push_back(neigh);
    }

    // Construct flipped_nodes_[i][k] = l such that
    // input site i contributes to output site l via kernel(k)
    for (int i = 0; i < nv_; ++i) {
      std::vector<int> flippednodes;
      Eigen::VectorXi coord = Site2Coord(i, lin_);
      for (auto t : trans) {
        Eigen::VectorXi newcoord(dim_);
        for (int d = 0; d < dim_; ++d) {
          newcoord(d) = ((coord(d) - t(d)) % lin_ + lin_) % lin_;
        }
        Eigen::VectorXi newcoordprime = newcoord / stride_;
        if (newcoordprime * stride_ == newcoord) {
          flippednodes.push_back(Coord2Site(newcoordprime, lout_));
        } else {
          flippednodes.push_back(-1);
        }
      }
      flipped_nodes_.push_back(flippednodes);
    }

    kernels_.resize(in_channels_ * kernel_size_, out_channels_);
    bias_.resize(out_channels_);

    lowered_image_.resize(in_channels_ * kernel_size_, nout_);
    lowered_image2_.resize(nout_, in_channels_ * kernel_size_);
    lowered_der_.resize(kernel_size_ * out_channels_, nv_);
    flipped_kernels_.resize(kernel_size_ * out_channels_, in_channels_);

    npar_ = in_channels_ * kernel_size_ * out_channels_;

    if (usebias_) {
      npar_ += out_channels_;
    } else {
      bias_.setZero();
    }

    name_ = "Convolutional Hypercube Layer";
  }

  std::string Name() const override { return name_; }

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
      for (int i = 0; i < out_channels_; ++i) {
        pars(k) = bias_(i);
        ++k;
      }
    }

    for (int j = 0; j < out_channels_; ++j) {
      for (int i = 0; i < in_channels_ * kernel_size_; ++i) {
        pars(k) = kernels_(i, j);
        ++k;
      }
    }
  }

  void SetParameters(VectorConstRefType pars) override {
    int k = 0;

    if (usebias_) {
      for (int i = 0; i < out_channels_; ++i) {
        bias_(i) = pars(k);
        ++k;
      }
    }

    for (int j = 0; j < out_channels_; ++j) {
      for (int i = 0; i < in_channels_ * kernel_size_; ++i) {
        kernels_(i, j) = pars(k);
        ++k;
      }
    }
  }

  void UpdateLookup(const VectorType &input,
                    const std::vector<int> &input_changes,
                    const VectorType &new_input, const VectorType &output,
                    std::vector<int> &output_changes,
                    VectorType &new_output) override {
    // At the moment the light cone structure of the convolution is not
    // exploited. To do so we would to change the part
    // else if (num_of_changes >0) {...}
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
    Convolve(input, output);

    if (usebias_) {
      int k = 0;
      for (int out = 0; out < out_channels_; ++out) {
        for (int i = 0; i < nout_; ++i) {
          output(k) += bias_(out);
          ++k;
        }
      }
    };
  }

  // performs the convolution of the kernel onto the image and writes into z
  inline void Convolve(const VectorType &image, VectorType &z) {
    // im2col method
    for (int i = 0; i < nout_; ++i) {
      int j = 0;
      for (auto n : neighbours_[i]) {
        for (int in = 0; in < in_channels_; ++in) {
          lowered_image_(in * kernel_size_ + j, i) = image(in * nv_ + n);
        }
        j++;
      }
    }
    Eigen::Map<MatrixType> output_image(z.data(), nout_, out_channels_);
    output_image.noalias() = lowered_image_.transpose() * kernels_;
  }

  inline void UpdateOutput(const VectorType &v,
                           const std::vector<int> &input_changes,
                           const VectorType &new_input,
                           VectorType &new_output) {
    const int num_of_changes = input_changes.size();
    for (int s = 0; s < num_of_changes; ++s) {
      const int sf = input_changes[s];
      int kout = 0;
      for (int out = 0; out < out_channels_; ++out) {
        for (int k = 0; k < kernel_size_; ++k) {
          if (flipped_nodes_[sf][k] >= 0) {
            new_output(flipped_nodes_[sf][k] + kout) +=
                kernels_(k, out) * (new_input(s) - v(sf));
          }
        }
        kout += nout_;
      }
    }
  }

  void Backprop(const VectorType &prev_layer_output,
                const VectorType & /*this_layer_output*/,
                const VectorType &dout, VectorType &din,
                VectorRefType der) override {
    // VectorType dLz = dout;
    int kd = 0;

    // Derivative for bias, d(L) / d(b) = d(L) / d(z)
    if (usebias_) {
      int k = 0;
      for (int out = 0; out < out_channels_; ++out) {
        der(kd) = 0;
        for (int i = 0; i < nout_; ++i) {
          der(kd) += dout(k);
          ++k;
        }
        ++kd;
      }
    }

    // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
    // Reshape dLdZ
    Eigen::Map<const MatrixType> dLz_reshaped(dout.data(), nout_,
                                              out_channels_);

    // Reshape image
    for (int in = 0; in < in_channels_; ++in) {
      for (int k = 0; k < kernel_size_; ++k) {
        for (int i = 0; i < nout_; ++i) {
          lowered_image2_(i, k + in * kernel_size_) =
              prev_layer_output(in * nv_ + neighbours_[i][k]);
        }
      }
    }
    Eigen::Map<MatrixType> der_w(der.data() + kd, in_channels_ * kernel_size_,
                                 out_channels_);
    der_w.noalias() = lowered_image2_.transpose() * dLz_reshaped;

    // Compute d(L) / d_in = W * [d(L) / d(z)]
    // int kout = 0;
    for (int out = 0; out < out_channels_; ++out) {
      for (int in = 0; in < in_channels_; ++in) {
        flipped_kernels_.block(out * kernel_size_, in, kernel_size_, 1) =
            kernels_.block(in * kernel_size_, out, kernel_size_, 1);
      }
    }

    for (int i = 0; i < nv_; i++) {
      int j = 0;
      for (auto n : flipped_nodes_[i]) {
        for (int out = 0; out < out_channels_; ++out) {
          lowered_der_(out * kernel_size_ + j, i) =
              n >= 0 ? dout(out * nout_ + n) : 0;
        }
        j++;
      }
    }

    din.resize(in_size_);
    Eigen::Map<MatrixType> der_in(din.data(), nv_, in_channels_);
    der_in.noalias() = lowered_der_.transpose() * flipped_kernels_;
  }

  void to_json(json &pars) const override {
    json layerpar;
    layerpar["Name"] = "Convolutional";
    layerpar["UseBias"] = usebias_;
    layerpar["Inputs"] = in_size_;
    layerpar["Outputs"] = out_size_;
    layerpar["InputChannels"] = in_channels_;
    layerpar["OutputChannels"] = out_channels_;
    layerpar["Bias"] = bias_;
    layerpar["Kernels"] = kernels_;

    pars["Layers"].push_back(layerpar);
  }

  void from_json(const json &pars) override {
    if (FieldExists(pars, "Kernels")) {
      kernels_ = pars["Kernels"];
    } else {
      kernels_.setZero();
    }
    if (FieldExists(pars, "Bias")) {
      bias_ = pars["Bias"];
    } else {
      bias_.setZero();
    }
  }

  int Coord2Site(Eigen::VectorXi const &coord, int L) {
    auto site = 0;
    auto scale = 1;
    for (int i = 0; i < dim_; ++i) {
      site += scale * coord(i);
      scale *= L;
    }
    return site;
  }

  Eigen::VectorXi Site2Coord(int site, int L) {
    Eigen::VectorXi coord(dim_);
    for (int i = 0; i < dim_; ++i) {
      coord(i) = site % L;
      site /= L;
    }
    return coord;
  }
};
}  // namespace netket

#endif
